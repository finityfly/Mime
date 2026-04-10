from __future__ import annotations

import argparse
import os
import time
from threading import Event, Thread

import cv2
import numpy as np
from dotenv import load_dotenv
from huggingface_hub import login
import pyaudio

from utils.media_bridge import ThreadedZoomBridge, ZoomMediaConfig
from utils.video_client import VideoClient
from utils.inference_engine_fast import InferenceEngineFast
from utils.arkit_renderer import ARKitRenderer
from utils.mt_engine import MTProcessor
from utils.stt_engine import STTProcessor
from utils.tts_engine import TTSProcessor


class MimeClient:
    """Unified client launcher for STT -> MT -> TTS and ARKit Avatar Zoom sinks."""

    def __init__(self, args: argparse.Namespace):
        from queue import Queue

        self.args = args
        self.stt_to_mt = Queue()
        self.mt_to_tts = Queue()

        self.tts: TTSProcessor | None = None
        self.mt: MTProcessor | None = None
        self.stt: STTProcessor | None = None

        self.inference_engine: InferenceEngine | None = None
        self.renderer: ARKitRenderer | None = None
        self.bridge: ThreadedZoomBridge | None = None

    def logger(self, message: str) -> None:
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] {message}")

    def _start_zoom_bridge(self) -> None:
        """Initializes the visual and audio bridge for Zoom."""
        # 1. Start the Inference Engine (The AI logic)
        self.inference_engine = InferenceEngineFast(
            model_path=self.args.abs_model,
            log_callback=self.logger,
            sample_rate=self.args.audio_rate,
        )

        # 2. Start the Renderer (The 3D visualization)
        self.renderer = ARKitRenderer(
            glb_path=self.args.avatar_model,
            engine=self.inference_engine,
            width=self.args.zoom_width,
            height=self.args.zoom_height
        )

        # 3. Setup the Media Bridge
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

        # 4. Setup VideoClient in 'abs' mode to pull from the Renderer
        self._video_client = VideoClient(
            width=self.args.zoom_width, 
            height=self.args.zoom_height, 
            fps=self.args.zoom_fps, 
            log=self.logger
        )

        if self.args.mode == "avatar":
            self.logger("[Client] Setting VideoClient to ARKit Avatar mode.")
            self._video_client.set_mode("abs")
            self._video_client.set_renderer(self.renderer.render_frame)
        elif self.args.forward_webcam:
            self._video_client.set_mode("webcam")
            self._video_client.set_webcam_index(self.args.webcam_index)
        else:
            self._video_client.set_mode("text")
            self._video_client.set_text("MIME CLIENT ONLINE\nWaiting for Audio...")

        try:
            self.bridge.set_video_source(self._video_client)
        except Exception as e:
            self.logger(f"[Client] Failed to set VideoClient: {e}")

        self.bridge.start()
        self.logger("[Client] Zoom bridge enabled.")

    def start(self) -> None:
        self.logger("Initializing Mime client...")

        def choose_audio_input_device(logger) -> int | None:
            pa = pyaudio.PyAudio()
            devices = []
            for i in range(pa.get_device_count()):
                info = pa.get_device_info_by_index(i)
                if int(info.get("maxInputChannels", 0)) > 0:
                    devices.append((i, info.get("name", f"Device {i}")))

            logger("[Client] Available audio input devices:")
            for idx, name in devices:
                logger(f"  {idx}: {name}")

            if self.args.audio_input_index is not None:
                return self.args.audio_input_index

            try:
                import sys as _sys
                if not _sys.stdin.isatty():
                    return None
                choice = input("Select audio input device index (blank for default): ").strip()
                return int(choice) if choice else None
            except Exception:
                return None

        selected_input_index = choose_audio_input_device(self.logger)

        if self.args.debug_audio:
            self.logger("[Client] Debug audio mode enabled: Zoom bridge disabled, TTS routed to local speakers.")
            # No callback means TTSProcessor uses local PyAudio playback.
            self.tts = TTSProcessor(self.mt_to_tts, self.logger)
        else:
            self._start_zoom_bridge()

            # DUAL AUDIO SINK: Redirect TTS output to both Zoom and the Animation Engine
            def dual_audio_callback(audio_bytes: bytes):
                # 1. Send to Zoom Virtual Cable
                if self.bridge:
                    self.bridge.push_audio(audio_bytes)
                # 2. Send to Inference Engine for lip-sync
                if self.inference_engine:
                    self.inference_engine.process_audio_chunk(audio_bytes)

            self.tts = TTSProcessor(self.mt_to_tts, self.logger, audio_callback=dual_audio_callback)

        self.mt = MTProcessor(self.stt_to_mt, self.mt_to_tts, self.logger)
        self.stt = STTProcessor(self.stt_to_mt, self.logger, input_device_index=selected_input_index)

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
        self.logger("[Client] Shutting down modules...")
        for module in [self.stt, self.mt, self.tts]:
            if module:
                try:
                    module.stop()
                except Exception:
                    pass

        if getattr(self, "_video_client", None):
            self._video_client.close()

        if self.bridge:
            self.bridge.stop()

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch Mime client services.")
    # Core Bridge Settings
    parser.add_argument("--debug-audio", action="store_true", help="Enable debug audio output.")
    parser.add_argument("--mode", type=str, default="avatar", choices=["avatar", "webcam", "text"])
    
    # Model Paths
    parser.add_argument("--abs-model", type=str, default="models/best_fast.pt", help="Path to .pt model.")
    parser.add_argument("--avatar-model", type=str, default="assets/avatar.glb", help="Path to .glb rig.")

    # Video Settings
    parser.add_argument("--zoom-width", type=int, default=1280)
    parser.add_argument("--zoom-height", type=int, default=720)
    parser.add_argument("--zoom-fps", type=int, default=30)

    # Audio Settings
    parser.add_argument("--audio-rate", type=int, default=48000)
    parser.add_argument("--audio-buffer", type=int, default=960)
    parser.add_argument("--audio-device-name", type=str, default=None)
    parser.add_argument("--audio-input-index", type=int, default=None)

    # Legacy Webcam Support
    parser.add_argument("--forward-webcam", action="store_true", default=False)
    parser.add_argument("--webcam-index", type=int, default=0)

    return parser.parse_args()

def main() -> None:
    load_dotenv()
    if os.getenv("HF_TOKEN"):
        login(token=os.getenv("HF_TOKEN"))
    
    args = parse_args()
    client = MimeClient(args)
    client.start()

if __name__ == "__main__":
    main()