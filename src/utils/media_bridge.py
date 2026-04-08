from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from queue import Empty, Queue
from threading import Thread
from typing import Callable, Optional

import cv2
import numpy as np
import pyaudio
import pyvirtualcam


@dataclass
class ZoomMediaConfig:
    """Configuration for publishing generated media into Zoom-compatible virtual devices."""

    width: int = 1280
    height: int = 720
    fps: int = 30
    camera_backend: Optional[str] = None

    audio_sample_rate: int = 48000
    audio_channels: int = 1
    audio_frames_per_buffer: int = 960
    audio_device_index: Optional[int] = None
    audio_device_name_contains: Optional[str] = None

    video_queue_size: int = 4
    audio_queue_size: int = 128
    reconnect_interval_sec: float = 3.0


class VideoFrameSource:
    """Optional pull-source interface for generated BGR frames."""

    def read(self) -> Optional[np.ndarray]:
        raise NotImplementedError("VideoFrameSource.read() is a placeholder.")


class AudioChunkSource:
    """Optional pull-source interface for generated audio chunks."""

    def read(self) -> Optional[np.ndarray | bytes]:
        raise NotImplementedError("AudioChunkSource.read() is a placeholder.")


class ZoomMediaBridge:
    """
    Production transport bridge for Zoom virtual camera + virtual cable audio.

    The bridge is sink-oriented and decoupled with internal queues:
    - push_frame(frame_bgr) for generated BGR frames
    - push_audio(chunk) for generated Float32/Int16/bytes audio
    """

    def __init__(
        self,
        config: ZoomMediaConfig,
        video_source: Optional[VideoFrameSource] = None,
        audio_source: Optional[AudioChunkSource] = None,
        log_callback: Callable[[str], None] = print,
    ):
        self.config = config
        self.video_source = video_source
        self.audio_source = audio_source
        self.log = log_callback
        self.is_running = False

        self._video_q: Queue[np.ndarray] = Queue(maxsize=max(1, config.video_queue_size))
        self._audio_q: Queue[np.ndarray | bytes] = Queue(maxsize=max(1, config.audio_queue_size))

        self._pa: Optional[pyaudio.PyAudio] = None
        self._audio_stream = None
        self._camera: Optional[pyvirtualcam.Camera] = None

        self._source_video_thread: Optional[Thread] = None
        self._source_audio_thread: Optional[Thread] = None

        self._last_frame_bgr = np.zeros((config.height, config.width, 3), dtype=np.uint8)
        self._next_cam_retry_ts = 0.0
        self._next_audio_retry_ts = 0.0

    def set_video_source(self, video_source: VideoFrameSource) -> None:
        self.video_source = video_source

    def set_audio_source(self, audio_source: AudioChunkSource) -> None:
        self.audio_source = audio_source

    def push_frame(self, frame: np.ndarray) -> None:
        if frame is None:
            return
        self._put_latest(self._video_q, frame)

    def push_audio(self, chunk: np.ndarray | bytes) -> None:
        if chunk is None:
            return
        self._put_latest(self._audio_q, chunk)

    def list_audio_output_devices(self) -> list[tuple[int, str]]:
        pa = self._ensure_pyaudio()
        devices: list[tuple[int, str]] = []
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            if int(info.get("maxOutputChannels", 0)) > 0:
                devices.append((i, str(info.get("name", f"Device {i}"))))
        return devices

    def start(self) -> None:
        self.is_running = True
        self.log("[ZoomBridge] Starting bridge transport loop.")
        self._start_source_threads()

        def _audio_worker() -> None:
            while self.is_running:
                now = time.time()
                self._ensure_audio_sink(now)
                self._process_audio_once()
                time.sleep(0.001)

        self._audio_thread = Thread(target=_audio_worker, daemon=True)
        self._audio_thread.start()

        while self.is_running:
            now = time.time()
            self._ensure_video_sink(now)
            self._process_video_once()
            time.sleep(0.001)

        if getattr(self, "_audio_thread", None) is not None:
            self._audio_thread.join(timeout=1.0)
        self._close_camera()
        self._close_audio_stream()
        self._close_pyaudio()
        self.log("[ZoomBridge] Bridge stopped.")

    def stop(self) -> None:
        self.is_running = False

    def _put_latest(self, q: Queue, item) -> None:
        try:
            q.put_nowait(item)
        except Exception:
            try:
                q.get_nowait()
            except Empty:
                pass
            try:
                q.put_nowait(item)
            except Exception:
                pass

    def _start_source_threads(self) -> None:
        if self.video_source is not None and self._source_video_thread is None:
            self._source_video_thread = Thread(target=self._video_source_loop, daemon=True)
            self._source_video_thread.start()

        if self.audio_source is not None and self._source_audio_thread is None:
            self._source_audio_thread = Thread(target=self._audio_source_loop, daemon=True)
            self._source_audio_thread.start()

    def _video_source_loop(self) -> None:
        assert self.video_source is not None
        while self.is_running:
            try:
                frame = self.video_source.read()
                if frame is not None:
                    self.push_frame(frame)
                else:
                    time.sleep(0.002)
            except Exception as exc:
                self.log(f"[ZoomBridge] Video source error: {exc}")
                time.sleep(0.05)

    def _audio_source_loop(self) -> None:
        assert self.audio_source is not None
        while self.is_running:
            try:
                chunk = self.audio_source.read()
                if chunk is not None:
                    self.push_audio(chunk)
                else:
                    time.sleep(0.002)
            except Exception as exc:
                self.log(f"[ZoomBridge] Audio source error: {exc}")
                time.sleep(0.05)

    def _ensure_video_sink(self, now_ts: float) -> None:
        if self._camera is not None or now_ts < self._next_cam_retry_ts:
            return

        try:
            self._camera = pyvirtualcam.Camera(
                width=self.config.width,
                height=self.config.height,
                fps=self.config.fps,
                backend=self.config.camera_backend,
            )
            self.log(f"[ZoomBridge] Virtual camera connected: {self._camera.device}")
        except Exception as exc:
            self._camera = None
            self._next_cam_retry_ts = now_ts + self.config.reconnect_interval_sec
            self.log(
                "[ZoomBridge] Virtual camera unavailable; retrying in "
                f"{self.config.reconnect_interval_sec:.0f}s ({exc})"
            )

    def _process_video_once(self) -> None:
        if self._camera is None:
            return

        latest = None
        while True:
            try:
                latest = self._video_q.get_nowait()
            except Empty:
                break

        if latest is not None:
            self._last_frame_bgr = self._fit_frame_letterbox(latest)

        try:
            self._camera.send(cv2.cvtColor(self._last_frame_bgr, cv2.COLOR_BGR2RGB))
            self._camera.sleep_until_next_frame()
        except Exception as exc:
            self.log(f"[ZoomBridge] Virtual camera write failed ({exc}); reconnecting.")
            self._close_camera()
            self._next_cam_retry_ts = time.time() + self.config.reconnect_interval_sec

    def _fit_frame_letterbox(self, frame: np.ndarray) -> np.ndarray:
        if frame is None or frame.size == 0:
            return self._last_frame_bgr

        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.ndim == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        src_h, src_w = frame.shape[:2]
        dst_w, dst_h = self.config.width, self.config.height
        if src_w <= 0 or src_h <= 0:
            return self._last_frame_bgr

        scale = min(dst_w / src_w, dst_h / src_h)
        out_w = max(1, int(src_w * scale))
        out_h = max(1, int(src_h * scale))
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        resized = cv2.resize(frame, (out_w, out_h), interpolation=interp)

        canvas = np.zeros((dst_h, dst_w, 3), dtype=np.uint8)
        x = (dst_w - out_w) // 2
        y = (dst_h - out_h) // 2
        canvas[y : y + out_h, x : x + out_w] = resized
        return canvas

    def _ensure_audio_sink(self, now_ts: float) -> None:
        if self._audio_stream is not None or now_ts < self._next_audio_retry_ts:
            return

        try:
            self._audio_stream = self._open_audio_output_stream()
            self.log("[ZoomBridge] Virtual cable audio sink connected.")
        except Exception as exc:
            self._audio_stream = None
            self._next_audio_retry_ts = now_ts + self.config.reconnect_interval_sec
            self.log(
                "[ZoomBridge] Audio sink unavailable; retrying in "
                f"{self.config.reconnect_interval_sec:.0f}s ({exc})"
            )

    def _resolve_audio_device_index(self) -> Optional[int]:
        if self.config.audio_device_index is not None:
            return self.config.audio_device_index

        devices = self.list_audio_output_devices()
        needle = self.config.audio_device_name_contains

        if needle:
            needle_l = needle.lower()
            for idx, name in devices:
                if needle_l in name.lower():
                    self.log(f"[ZoomBridge] Selected audio output device: {name} (index={idx})")
                    return idx
            raise RuntimeError(
                f"No output audio device matched '{needle}'. "
                "Use list_audio_output_devices() to inspect devices."
            )

        # Windows-first auto-discovery for VB-CABLE style routing.
        preferred = ["cable input", "vb-audio", "vb cable", "virtual cable", "cable"]
        for keyword in preferred:
            for idx, name in devices:
                if keyword in name.lower():
                    self.log(f"[ZoomBridge] Auto-selected virtual cable device: {name} (index={idx})")
                    return idx

        if sys.platform.startswith("win"):
            raise RuntimeError(
                "No VB-CABLE output device found. Install VB-CABLE and ensure device names "
                "contain 'CABLE Input' or 'VB-Audio'."
            )

        self.log("[ZoomBridge] No virtual cable pattern matched; using default output device.")
        return None

    def _open_audio_output_stream(self):
        pa = self._ensure_pyaudio()
        out_device = self._resolve_audio_device_index()
        return pa.open(
            format=pyaudio.paFloat32,
            channels=self.config.audio_channels,
            rate=self.config.audio_sample_rate,
            output=True,
            output_device_index=out_device,
            frames_per_buffer=self.config.audio_frames_per_buffer,
        )

    def _normalize_audio_chunk(self, chunk: np.ndarray | bytes) -> np.ndarray:
        if isinstance(chunk, bytes):
            arr = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
        else:
            arr = np.asarray(chunk)
            if arr.dtype == np.int16:
                arr = arr.astype(np.float32) / 32768.0
            elif arr.dtype != np.float32:
                arr = arr.astype(np.float32)

        if arr.size == 0:
            return np.zeros(0, dtype=np.float32)

        # Ensure shape is (frames, channels) so we can match config.audio_channels.
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)

        target_ch = self.config.audio_channels
        src_ch = arr.shape[1]
        if src_ch != target_ch:
            if target_ch == 1:
                # Downmix: average all source channels.
                arr = np.mean(arr, axis=1, keepdims=True)
            else:
                # Downmix to mono first, then spread to every target channel.
                mono = np.mean(arr, axis=1, keepdims=True)
                arr = np.repeat(mono, target_ch, axis=1)

        peak = float(np.max(np.abs(arr)))
        if peak > 1.0:
            arr *= 1.0 / peak
        elif peak > 0.98:
            arr *= 0.98 / peak

        # Soft limiter to reduce clipping artifacts during loud TTS bursts.
        drive = 1.35
        arr = np.tanh(arr * drive) / np.tanh(drive)
        arr = np.clip(arr, -1.0, 1.0).astype(np.float32, copy=False)

        # Return interleaved samples (C-contiguous) for PyAudio.
        return np.ascontiguousarray(arr).reshape(-1)

    def _process_audio_once(self) -> None:
        if self._audio_stream is None:
            return

        chunk = None
        try:
            chunk = self._audio_q.get_nowait()
        except Empty:
            return

        try:
            normalized = self._normalize_audio_chunk(chunk)
            self._audio_stream.write(normalized.tobytes())
        except Exception as exc:
            self.log(f"[ZoomBridge] Audio write failed ({exc}); reconnecting.")
            self._close_audio_stream()
            self._next_audio_retry_ts = time.time() + self.config.reconnect_interval_sec

    def _ensure_pyaudio(self) -> pyaudio.PyAudio:
        if self._pa is None:
            self._pa = pyaudio.PyAudio()
        return self._pa

    def _close_camera(self) -> None:
        if self._camera is None:
            return
        try:
            self._camera.close()
        except Exception:
            pass
        self._camera = None

    def _close_audio_stream(self) -> None:
        if self._audio_stream is None:
            return
        try:
            self._audio_stream.stop_stream()
            self._audio_stream.close()
        except Exception:
            pass
        self._audio_stream = None

    def _close_pyaudio(self) -> None:
        if self._pa is None:
            return
        try:
            self._pa.terminate()
        except Exception:
            pass
        self._pa = None


class ThreadedZoomBridge(ZoomMediaBridge):
    """ZoomMediaBridge variant that owns and manages its run thread."""

    def __init__(
        self,
        config: ZoomMediaConfig,
        video_source: Optional[VideoFrameSource] = None,
        audio_source: Optional[AudioChunkSource] = None,
        log_callback: Callable[[str], None] = print,
    ):
        super().__init__(
            config=config,
            video_source=video_source,
            audio_source=audio_source,
            log_callback=log_callback,
        )
        self._run_thread: Optional[Thread] = None

    def start(self) -> None:
        if self._run_thread is not None and self._run_thread.is_alive():
            return
        self._run_thread = Thread(target=super().start, daemon=True)
        self._run_thread.start()

    def stop(self) -> None:
        super().stop()
        if self._run_thread is not None:
            self._run_thread.join(timeout=2.0)
            self._run_thread = None
