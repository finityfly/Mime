import os
import io
import wave
import queue
import threading
import time
import pyaudio
import numpy as np
from groq import Groq
import string


class STTProcessor:
    def __init__(self, output_queue, log_callback, input_device_index: int | None = None):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.output_queue = output_queue
        self.log = log_callback
        self.is_running = True

        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 16000

        # Placeholder -- will be set during calibrate_noise_floor().
        # The adaptive threshold replaces the old hard-coded value of 800,
        # which was far too low for typical ambient noise levels.
        self.SILENCE_THRESHOLD = 800
        self.MAX_SILENT_CHUNKS = 15   # ~15 * 1024/16000 = ~0.96s of silence to finalise
        self.MIN_SPEECH_CHUNKS = 8    # ~0.5s minimum utterance -- eliminates clicks/pops

        # Expanded list of phrases commonly hallucinated by Whisper from noise.
        # These are reliably returned by the model even when no speech occurred.
        self.HALLUCINATIONS = [
            "thank you", "thanks for watching", "thanks",
            "subtitle", "subtitles", "caption", "captions",
            "bye bye", "goodbye", "see you", "see ya",
            "music", "music playing", "background music",
            "applause", "laughter", "cheering",
            "um", "uh", "hmm", "mm-hmm", "mm", "mhm",
            "you", "the", "a", "and", "to", "of", "in", "it", "is",
            "i'm sorry", "sorry", "excuse me",
            "foreign", "foreign language",
            "silence", "silent", "quiet",
            "speaker", "unknown speaker", "inaudible",
        ]

        self.p = pyaudio.PyAudio()
        self.input_device_index = input_device_index
        if self.input_device_index is not None:
            self.log(f"[STT] Using audio input device index: {self.input_device_index}")
        self.audio_queue = queue.Queue()
        self._stream = None
        self._stream_thread = None
        self._process_thread = None

    def _calculate_rms(self, frame):
        """Compute RMS amplitude of a 16-bit PCM frame."""
        data = np.frombuffer(frame, dtype=np.int16)
        if len(data) == 0:
            return 0
        return np.sqrt(np.mean(data.astype(np.float64) ** 2))

    def _calibrate_noise_floor(self):
        """
        Sample 1 second of ambient audio and set an adaptive silence threshold.

        The threshold is set to max(mean_rms x 3.5, 800) so it scales with
        the user's environment while never dropping below a sane minimum.
        A quiet room might calibrate to ~800-1200; a noisier space to higher.
        """
        self.log("[STT] Calibrating noise floor -- please remain quiet for 1 second...")
        cal_stream = None
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

            cal_stream = self.p.open(**open_kwargs)
            rms_values = []
            samples_needed = int(self.RATE / self.CHUNK)  # ~16 chunks for 1s

            for _ in range(samples_needed):
                data = cal_stream.read(self.CHUNK, exception_on_overflow=False)
                rms_values.append(self._calculate_rms(data))

            mean_rms = np.mean(rms_values) if rms_values else 0
            adaptive_threshold = int(mean_rms * 3.5)
            self.SILENCE_THRESHOLD = max(adaptive_threshold, 800)
            self.log(
                f"[STT] Noise floor calibrated: mean_rms={mean_rms:.1f}, "
                f"threshold set to {self.SILENCE_THRESHOLD}"
            )
        except Exception as e:
            self.log(f"[STT] Noise calibration failed ({e}), using default threshold {self.SILENCE_THRESHOLD}")
        finally:
            if cal_stream is not None:
                try:
                    cal_stream.stop_stream()
                    cal_stream.close()
                except Exception:
                    pass

    def _is_hallucination(self, text: str) -> bool:
        """
        Return True if text looks like a Whisper noise hallucination rather than real speech.
        """
        lower = text.lower().strip()

        # Reject single characters
        if len(lower) < 2:
            return True

        # Reject very short fragments (fewer than 3 chars) unless they contain
        # meaningful alphabetic content like "ok" or "hi"
        if len(lower) < 3 and not lower.isalpha():
            return True

        # Normalise: strip common trailing punctuation so "thank you." and "bye bye!"
        # match the exact entries in HALLUCINATIONS without broadening the filter.
        stripped = lower.rstrip(string.punctuation)

        # Check the expanded hallucination phrase list -- exact match only to avoid
        # rejecting valid speech like "a quick test" or "to be or not to be"
        for phrase in self.HALLUCINATIONS:
            if phrase == lower or phrase == stripped:
                return True

        # Reject text with no alphabetic characters (pure numbers, punctuation, symbols)
        # Real speech always contains at least some letters.
        if not any(c.isalpha() for c in lower):
            return True

        # Reject text that's just repeated single characters (e.g., "aaa", "...")
        unique_chars = set(lower.replace(" ", ""))
        if len(unique_chars) <= 1 and len(lower) > 1:
            return True

        return False

    def _stream_audio(self):
        # Run noise floor calibration before starting the main listen loop
        self._calibrate_noise_floor()

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
            speech_chunks_count = 0
            recording_started = False

            while self.is_running:
                data = self._stream.read(self.CHUNK, exception_on_overflow=False)
                rms = self._calculate_rms(data)

                if rms > self.SILENCE_THRESHOLD:
                    if not recording_started:
                        recording_started = True
                    frames.append(data)
                    speech_chunks_count += 1
                    silent_chunks_count = 0
                else:
                    if recording_started:
                        frames.append(data)
                        silent_chunks_count += 1

                        if silent_chunks_count > self.MAX_SILENT_CHUNKS:
                            # Only queue if utterance was long enough to be real speech.
                            # Use speech_chunks_count to avoid counting trailing silence.
                            if speech_chunks_count >= self.MIN_SPEECH_CHUNKS:
                                self.audio_queue.put(frames)
                            frames = []
                            silent_chunks_count = 0
                            speech_chunks_count = 0
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

                # Comprehensive hallucination filter -- catches both known phrases
                # and structurally invalid transcriptions (too short, no letters, etc.)
                if not self._is_hallucination(text):
                    self.log(f"[STT] Heard: {text}")
                    self.output_queue.put(text)
                else:
                    self.log(f"[STT] Filtered hallucination: '{text}'")

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
        self._thread.start()
        self._process_thread.start()
