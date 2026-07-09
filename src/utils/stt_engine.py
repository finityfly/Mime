import os
import io
import wave
import queue
import threading
import pyaudio
import numpy as np
from groq import Groq

class STTProcessor:
    def __init__(self, output_queue, log_callback, input_device_index: int | None = None, playback_event=None):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.output_queue = output_queue
        self.log = log_callback
        self.is_running = True

        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000

        # calibration: raised threshold to avoid keyboard/typing noise being
        # detected as speech. MIN_SPEECH_CHUNKS prevents very short captures
        # (like clicks) from being sent to the recognizer.
        self.SILENCE_THRESHOLD = 800
        self.MAX_SILENT_CHUNKS = 12  # ~12 * 1024/16000 = ~0.77s of silence to finalize
        self.MIN_SPEECH_CHUNKS = 3   # ignore utterances shorter than ~0.19s

        self.p = pyaudio.PyAudio()
        self.input_device_index = input_device_index
        self.playback_event = playback_event
        if self.input_device_index is not None:
            self.log(f"[STT] Using audio input device index: {self.input_device_index}")
        self.audio_queue = queue.Queue()
        self._stream = None
        self._stream_thread = None
        self._process_thread = None

    def _calculate_rms(self, frame):
        data = np.frombuffer(frame, dtype=np.int16)
        if len(data) == 0:
            return 0
        return np.sqrt(np.mean(data.astype(np.float64)**2))

    def _stream_audio(self):
        try:
            open_kwargs = dict(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                frames_per_buffer=self.CHUNK,
            )
            if self.input_device_index is not None:
                open_kwargs["input_device_index"] = self.input_device_index

            self._stream = self.p.open(**open_kwargs)
            
            self.log("[STT] Mic active. Listening for full thoughts...")
            
            frames = []
            silent_chunks_count = 0
            recording_started = False

            while self.is_running:
                data = self._stream.read(self.CHUNK, exception_on_overflow=False)
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
                            # finalize only if speech was long enough
                            if len(frames) >= self.MIN_SPEECH_CHUNKS:
                                self.audio_queue.put(frames)
                            else:
                                pass
                            frames = []
                            silent_chunks_count = 0
                            recording_started = False
        
        except Exception as e:
            self.log(f"[STT] Hardware Error: {e}")
        finally:
            try:
                if self._stream is not None:
                    self._stream.stop_stream()
                    self._stream.close()
            except Exception:
                pass
            self._stream = None

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

    def stop(self, timeout: float = 1.0) -> None:
        """Gracefully stop the STT processor and join threads."""
        self.is_running = False
        # Wake processing thread if blocked
        try:
            self.audio_queue.put_nowait([])
        except Exception:
            pass

        if self._stream_thread is not None:
            self._stream_thread.join(timeout=timeout)
            self._stream_thread = None
        if self._process_thread is not None:
            self._process_thread.join(timeout=timeout)
            self._process_thread = None
        try:
            if self._stream is not None:
                self._stream.stop_stream()
                self._stream.close()
        except Exception:
            pass
        try:
            self.p.terminate()
        except Exception:
            pass

    def start(self):
        self._stream_thread = threading.Thread(target=self._stream_audio, daemon=True)
        self._process_thread = threading.Thread(target=self._process_audio, daemon=True)
        self._stream_thread.start()
        self._process_thread.start()