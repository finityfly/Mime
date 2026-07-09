import os, json, base64, asyncio, threading, time
import requests
import pyaudio


class TTSProcessor:
    def __init__(self, input_queue, log_callback, audio_callback=None, playback_event=None):
        self.input_queue = input_queue
        self.log = log_callback
        self.audio_callback = audio_callback
        self.auth = os.getenv("INWORLD_API_KEY")
        self.voice_id = os.getenv("INWORLD_VOICE_ID", "Étienne")
        self.model_id = os.getenv("INWORLD_TTS_MODEL", "inworld-tts-1.5-mini")
        self.language = os.getenv("INWORLD_TTS_LANGUAGE", "fr-FR")
        self.log(f"[TTS] Using voice='{self.voice_id}', model='{self.model_id}', language='{self.language}'")

        # Optional threading.Event (or similar) that is set while TTS is playing.
        self.playback_event = playback_event

        if self.audio_callback is None:
            self.p = pyaudio.PyAudio()
            self.stream = self.p.open(format=pyaudio.paInt16, channels=1, rate=48000, output=True)
        else:
            self.p = None
            self.stream = None

        # playback state helpers
        self._first_chunk = False
        # number of samples to ramp-in at chunk start to reduce clicks
        self._fade_samples = 2048

        # lifecycle
        self.is_running = False
        self._thread: threading.Thread | None = None



    def _prepare_audio(self, audio_bytes: bytes, is_first: bool = False) -> bytes:
        """Apply short fade-in on chunk starts to reduce ticking/clicking at discontinuities."""
        if not audio_bytes:
            return audio_bytes
        try:
            samples = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float64)
            if samples.size == 0:
                return audio_bytes
            n = min(self._fade_samples, samples.size)
            if n > 1 and is_first:
                ramp = np.linspace(0.0, 1.0, n)
                samples[:n] *= ramp
            # Mild end fade on short trailing silence edges reduces tick between sentences
            if n > 1 and samples.size >= n:
                end_ramp = np.linspace(1.0, 0.85, n)
                # only soften very last micro-edge, not full fade-out
                samples[-n:] *= end_ramp
            samples = np.clip(samples, -32768, 32767).astype(np.int16)
            return samples.tobytes()
        except Exception:
            return audio_bytes

    def _synthesize_one(self, text: str) -> None:
        headers = {"Content-Type": "application/json", "Authorization": f"Basic {self.auth}"}
        request_start = time.time()
        self._first_chunk = True

        # Sanitize text
        text = text.replace("\u2019", "'").replace("\u2018", "'")

        request_data = {
            "text": text,
            "voice_id": self.voice_id,
            "model_id": self.model_id,
            "audio_config": {
                "audio_encoding": "LINEAR16",
                "sample_rate_hertz": 48000,
            },
        }

        if self.language:
            request_data["language"] = self.language

        payload = request_data
        self.log(f"[TTS-DEBUG] Sending payload via HTTP streaming: {json.dumps(request_data, ensure_ascii=False)[:200]}")

        url = "https://api.inworld.ai/tts/v1/voice:stream"
        debug_buf = bytearray()
        received_first = False

        try:
            # mark playback active if user provided an event
            if self.playback_event is not None:
                try:
                    self.playback_event.set()
                except Exception:
                    pass
            # shorter timeout to avoid long blocking during shutdown
            with requests.post(url, headers=headers, json=payload, stream=True, timeout=30) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines(decode_unicode=True):
                    # allow quick abort if we're stopping
                    if not self.is_running:
                        self.log("[TTS] Aborting streaming due to stop request")
                        break
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        self.log(f"[TTS-DEBUG] Skipping non-JSON line")
                        continue

                    # Inworld may send audio nested under 'result' or top-level 'audioContent'
                    audio_b64 = None
                    if isinstance(chunk, dict):
                        if "result" in chunk and isinstance(chunk["result"], dict):
                            audio_b64 = chunk["result"].get("audioContent")
                        elif "audioContent" in chunk:
                            audio_b64 = chunk.get("audioContent")

                    if audio_b64:
                        audio_bytes = base64.b64decode(audio_b64)
                        if not received_first:
                            self.log(f"[TTS] First audio chunk latency: {time.time() - request_start:.2f}s")
                            received_first = True

                        if self.audio_callback is not None:
                            try:
                                self.audio_callback(audio_bytes)
                                self.log("[TTS-DEBUG] Forwarded chunk to audio_callback")
                            except Exception as e:
                                self.log(f"[TTS] audio_callback error: {e}")
                        else:
                            try:
                                self.stream.write(audio_bytes)
                            except Exception as e:
                                self.log(f"[TTS] Playback error: {e}")

                    # Some implementations may include an 'isFinal' flag in the chunk
                    if isinstance(chunk, dict) and chunk.get("isFinal"):
                        self.log("[TTS] Sentence finished (isFinal)")
                        break

        except requests.RequestException as e:
            self.log(f"[TTS] HTTP streaming error: {e}")
        except Exception as e:
            self.log(f"[TTS] Unexpected error during HTTP streaming: {type(e).__name__}: {e}")
        finally:
            # Clear playback event so STT can resume
            try:
                if self.playback_event is not None:
                    self.playback_event.clear()
            except Exception:
                pass

        # Streaming completed for this sentence.
        self.log("[TTS-DEBUG] Completed streaming for sentence")

    async def _run_loop(self) -> None:
        while self.is_running:
            text = await asyncio.to_thread(self.input_queue.get)
            # sentinel to stop
            if not isinstance(text, str) or not text.strip():
                if not self.is_running:
                    break
                continue

            self.log(f"[TTS] Requesting synthesis: '{text[:40]}...'")
            # _synthesize_one is a blocking HTTP streaming function; run it
            # in a thread to avoid blocking the asyncio event loop.
            await asyncio.to_thread(self._synthesize_one, text)

    def _start_async_loop(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._run_loop())
        try:
            loop.close()
        except Exception:
            pass

    def start(self) -> None:
        if self.is_running:
            return
        self.is_running = True
        self._thread = threading.Thread(target=self._start_async_loop, daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self.is_running = False
        try:
            self.input_queue.put_nowait("")
        except Exception:
            try:
                # fallback blocking put to ensure wake
                self.input_queue.put("")
            except Exception:
                pass

        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

        # Close playback resources
        try:
            if self.stream is not None:
                try:
                    self.stream.stop_stream()
                    self.stream.close()
                except Exception:
                    pass
                self.stream = None
        except Exception:
            pass
        try:
            if self.p is not None:
                self.p.terminate()
        except Exception:
            pass