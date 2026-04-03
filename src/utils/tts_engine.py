import os, json, base64, asyncio, threading, queue, pyaudio, time
import websockets

class TTSProcessor:
    def __init__(self, input_queue, log_callback):
        self.input_queue = input_queue
        self.log = log_callback
        self.auth = os.getenv("INWORLD_AUTH_SIGNATURE")
        self.ws_url = "wss://api.inworld.ai/v1/tts:synthesize-stream"
        
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=pyaudio.paInt16, channels=1, rate=48000, output=True)

    async def _ws_handler(self):
        headers = {"Authorization": f"Basic {self.auth}"}
        
        try:
            async with websockets.connect(
                self.ws_url, 
                additional_headers=headers
            ) as ws:
                self.log("[TTS] WebSocket Connected to Inworld")
                
                while True:
                    text = await asyncio.to_thread(self.input_queue.get)
                    if not text.strip(): continue

                    self.log(f"[TTS] Requesting Synthesis: '{text[:20]}...'")
                    request_start = time.time()

                    request = {
                        "text": text,
                        "voice_id": "Dennis",
                        "model_id": "inworld-tts-1.5-mini",
                        "audio_config": {"audio_encoding": "LINEAR16", "sample_rate_hertz": 48000}
                    }
                    
                    await ws.send(json.dumps(request))
                    
                    received_first_byte = False
                    async for message in ws:
                        data = json.loads(message)
                        
                        if "audioContent" in data:
                            if not received_first_byte:
                                ttfa = time.time() - request_start
                                self.log(f"[TTS-DEBUG] First audio chunk received (Latency: {ttfa:.2f}s)")
                                received_first_byte = True
                                
                            audio_bytes = base64.b64decode(data["audioContent"])
                            self.stream.write(audio_bytes)
                        
                        if data.get("isFinal"):
                            self.log("[TTS] Sentence Finished Playing")
                            break
                            
        except Exception as e:
            self.log(f"[TTS] WebSocket Error: {e}")

    def _start_async_loop(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self._ws_handler())

    def start(self):
        threading.Thread(target=self._start_async_loop, daemon=True).start()