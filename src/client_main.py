from __future__ import annotations

import argparse
import os
import time
from threading import Event, Thread

import cv2
import numpy as np
from dotenv import load_dotenv
from huggingface_hub import login

from utils.media_bridge import ThreadedZoomBridge, ZoomMediaConfig
from utils.mt_engine import MTProcessor
from utils.stt_engine import STTProcessor
from utils.tts_engine import TTSProcessor


class MimeClient:
    """Unified client launcher for STT -> MT -> TTS and optional Zoom sinks."""

    def __init__(self, args: argparse.Namespace):
        from queue import Queue

        self.args = args
        self.stt_to_mt = Queue()
        self.mt_to_tts = Queue()

        self.tts: TTSProcessor | None = None
        self.mt: MTProcessor | None = None
        self.stt: STTProcessor | None = None

        self.bridge: ThreadedZoomBridge | None = None
        self._frame_thread: Thread | None = None
        self._frame_stop = Event()

    def logger(self, message: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] {message}")

    def _make_startup_frame(self, width: int, height: int) -> np.ndarray:
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.rectangle(frame, (0, 0), (width, height), (18, 18, 18), thickness=-1)
        cv2.putText(frame, "MIME CLIENT ONLINE", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 220, 120), 3)
        cv2.putText(
            frame,
            "Waiting for MicABSMonitor frames...",
            (40, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (220, 220, 220),
            2,
        )
        cv2.putText(
            frame,
            "Bridge accepts push_frame(BGR) + push_audio(Int16/Float32/bytes)",
            (40, 170),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (200, 200, 200),
            2,
        )
        return frame

    def _bridge_frame_keepalive_loop(self) -> None:
        assert self.bridge is not None
        interval = 1.0 / max(1, int(self.args.zoom_fps))
        base = self._make_startup_frame(self.args.zoom_width, self.args.zoom_height)

        while not self._frame_stop.is_set():
            frame = base.copy()
            now = time.strftime("%H:%M:%S")
            cv2.putText(frame, f"{now}", (40, self.args.zoom_height - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 220), 2)
            self.bridge.push_frame(frame)
            self._frame_stop.wait(interval)

    def _start_zoom_bridge(self) -> None:
        self.bridge = ThreadedZoomBridge(
            config=ZoomMediaConfig(
                width=self.args.zoom_width,
                height=self.args.zoom_height,
                fps=self.args.zoom_fps,
                audio_sample_rate=self.args.audio_rate,
                audio_channels=1,
                audio_frames_per_buffer=self.args.audio_buffer,
                audio_device_name_contains=self.args.audio_device_name,
                reconnect_interval_sec=3.0,
            ),
            log_callback=self.logger,
        )
        self.bridge.start()

        self._frame_stop.clear()
        self._frame_thread = Thread(target=self._bridge_frame_keepalive_loop, daemon=True)
        self._frame_thread.start()

        self.logger("[Client] Zoom bridge enabled.")

    def start(self) -> None:
        self.logger("Initializing Mime client...")

        if self.args.enable_zoom_bridge:
            self._start_zoom_bridge()
            self.logger("[Client] In Zoom, select virtual camera + CABLE Output mic.")

        audio_callback = self.bridge.push_audio if self.bridge is not None else None
        self.tts = TTSProcessor(self.mt_to_tts, self.logger, audio_callback=audio_callback)
        self.mt = MTProcessor(self.stt_to_mt, self.mt_to_tts, self.logger)
        self.stt = STTProcessor(self.stt_to_mt, self.logger)

        self.tts.start()
        self.mt.start()
        self.stt.start()

        self.logger("SYSTEM ONLINE. Speak into your mic.")

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.logger("System shutting down...")
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        self._frame_stop.set()
        if self._frame_thread is not None:
            self._frame_thread.join(timeout=1.0)
            self._frame_thread = None

        if self.bridge is not None:
            self.bridge.stop()
            self.bridge = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch Mime client services.")
    parser.add_argument(
        "--enable-zoom-bridge",
        action="store_true",
        help="Start ThreadedZoomBridge (virtual camera + virtual cable audio sink).",
    )
    parser.add_argument("--zoom-width", type=int, default=1280, help="Virtual camera width.")
    parser.add_argument("--zoom-height", type=int, default=720, help="Virtual camera height.")
    parser.add_argument("--zoom-fps", type=int, default=30, help="Virtual camera FPS.")
    parser.add_argument("--audio-rate", type=int, default=48000, help="Bridge audio sample rate.")
    parser.add_argument("--audio-buffer", type=int, default=960, help="Bridge audio frames per buffer.")
    parser.add_argument(
        "--audio-device-name",
        type=str,
        default=None,
        help="Optional output device name substring override (for example: CABLE Input).",
    )
    return parser.parse_args()


def configure_env() -> None:
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        login(token=hf_token)


def main() -> None:
    configure_env()
    args = parse_args()
    client = MimeClient(args)
    client.start()


if __name__ == "__main__":
    main()
