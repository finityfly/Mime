import time
import os
from dotenv import load_dotenv
from queue import Queue
import threading
from utils.stt_engine import STTProcessor
from utils.mt_engine import MTProcessor
from utils.tts_engine import TTSProcessor
from huggingface_hub import login
import pyaudio

load_dotenv()
login(token=os.getenv("HF_TOKEN"))

class LiveOrchestrator:
    def __init__(self):
        self.stt_to_mt = Queue()
        self.mt_to_tts = Queue()
        self.playback_event = threading.Event()

    def logger(self, message):
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] {message}")

    def start_pipeline(self):
        self.logger("Initializing System...")
        # Allow interactive selection of the audio input device (microphone)
        def choose_audio_input_device(logger) -> int | None:
            pa = pyaudio.PyAudio()
            devices = []
            for i in range(pa.get_device_count()):
                info = pa.get_device_info_by_index(i)
                if int(info.get("maxInputChannels", 0)) > 0:
                    devices.append((i, info.get("name", f"Device {i}")))

            logger("[STS] Available audio input devices:")
            for idx, name in devices:
                logger(f"  {idx}: {name}")

            try:
                import sys as _sys
                if not _sys.stdin.isatty():
                    logger("[STS] stdin is not interactive; using default input device.")
                    return None
            except Exception:
                pass

            try:
                choice = input("Select audio input device index (blank for default): ").strip()
            except Exception:
                return None

            if choice == "":
                return None
            try:
                return int(choice)
            except Exception:
                logger("[STS] Invalid selection; using default input device.")
                return None

        selected_input_index = choose_audio_input_device(self.logger)

        self.tts = TTSProcessor(self.mt_to_tts, self.logger, playback_event=self.playback_event)
        self.mt = MTProcessor(self.stt_to_mt, self.mt_to_tts, self.logger)
        self.stt = STTProcessor(self.stt_to_mt, self.logger, input_device_index=selected_input_index, playback_event=self.playback_event)

        self.tts.start()
        self.mt.start()
        self.stt.start()

        self.logger("SYSTEM ONLINE. Speak into your mic.")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.logger("System shutting down...")

if __name__ == "__main__":
    app = LiveOrchestrator()
    app.start_pipeline()