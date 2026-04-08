import time
import os
from dotenv import load_dotenv
from queue import Queue
from utils.stt_engine import STTProcessor
from utils.mt_engine import MTProcessor
from utils.tts_engine import TTSProcessor
from huggingface_hub import login

load_dotenv()
login(token=os.getenv("HF_TOKEN"))

class LiveOrchestrator:
    def __init__(self):
        self.stt_to_mt = Queue()
        self.mt_to_tts = Queue()

    def logger(self, message):
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] {message}")

    def start_pipeline(self):
        self.logger("Initializing System...")
        
        self.tts = TTSProcessor(self.mt_to_tts, self.logger)
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

if __name__ == "__main__":
    app = LiveOrchestrator()
    app.start_pipeline()