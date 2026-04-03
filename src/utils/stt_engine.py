import os
import io
import wave
import queue
import threading
import pyaudio
import numpy as np
from groq import Groq

class STTProcessor:
    def __init__(self, output_queue, log_callback):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.output_queue = output_queue
        self.log = log_callback
        self.is_running = True
        
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000
        
        # calibration
        self.SILENCE_THRESHOLD = 200
        self.MAX_SILENT_CHUNKS = 8  # (16000/1024) * 1.2 seconds = ~18 chunks

        self.p = pyaudio.PyAudio()
        self.audio_queue = queue.Queue()

    def _calculate_rms(self, frame):
        data = np.frombuffer(frame, dtype=np.int16)
        if len(data) == 0:
            return 0
        return np.sqrt(np.mean(data.astype(np.float64)**2))

    def _stream_audio(self):
        try:
            stream = self.p.open(
                format=self.FORMAT, 
                channels=self.CHANNELS,
                rate=self.RATE, 
                input=True, 
                frames_per_buffer=self.CHUNK
            )
            
            self.log("[STT] Mic active. Listening for full thoughts...")
            
            frames = []
            silent_chunks_count = 0
            recording_started = False

            while self.is_running:
                data = stream.read(self.CHUNK, exception_on_overflow=False)
                rms = self._calculate_rms(data)
                
                if rms > self.SILENCE_THRESHOLD:
                    if not recording_started:
                        recording_started = True
                        # self.log("[STT-DEBUG] Speech started...")
                    frames.append(data)
                    silent_chunks_count = 0
                else:
                    if recording_started:
                        frames.append(data)
                        silent_chunks_count += 1
                        
                        if silent_chunks_count > self.MAX_SILENT_CHUNKS:
                            # self.log(f"[STT-DEBUG] Thought finalized ({len(frames)} chunks)")
                            self.audio_queue.put(frames)
                            frames = []
                            silent_chunks_count = 0
                            recording_started = False
        
        except Exception as e:
            self.log(f"[STT] Hardware Error: {e}")

    def _process_audio(self):
        while self.is_running:
            try:
                frames = self.audio_queue.get()
                
                buf = io.BytesIO()
                with wave.open(buf, 'wb') as wf:
                    wf.setnchannels(self.CHANNELS)
                    wf.setsampwidth(2)
                    wf.setframerate(self.RATE)
                    wf.writeframes(b''.join(frames))
                buf.seek(0)

                resp = self.client.audio.transcriptions.create(
                    file=("chunk.wav", buf),
                    model="whisper-large-v3-turbo",
                    response_format="text",
                    temperature=0.0
                )
                text = resp.strip()
                lower_text = text.lower()

                hallucinations = ["thank you", "thanks for watching", "subtitle", "bye bye"]
                if text and not any(h in lower_text for h in hallucinations):
                    if len(text) > 1: # Ignore single character dots/noises
                        self.log(f"[STT] Heard: {text}")
                        self.output_queue.put(text)
                    
            except Exception as e:
                self.log(f"[STT] API Error: {e}")

    def start(self):
        threading.Thread(target=self._stream_audio, daemon=True).start()
        threading.Thread(target=self._process_audio, daemon=True).start()