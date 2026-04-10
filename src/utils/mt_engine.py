import os, queue, threading, time
from groq import Groq

class MTProcessor:
    def __init__(self, input_queue, output_queue, log_callback):
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.log = log_callback
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model_id = "llama-3.3-70b-versatile"
        self.is_running = True
        self._thread = None
        
        # context buffer for gendered pronoun resolution and handling fragmented sentences
        self.context_history = [] 
        self.max_context = 3

    def _get_context_string(self):
        return " ".join(self.context_history)

    def _run(self):
        self.log("[MT] Groq API Translation Engine Online")
        while self.is_running:
            text = self.input_queue.get()
            # allow shutdown sentinel
            if not isinstance(text, str) or not text.strip():
                # If stopping, break; otherwise continue waiting
                if not self.is_running:
                    break
                continue

            start_time = time.time()
            context = self._get_context_string()

            try:
                completion = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=[
                        {
                            "role": "system", 
                            "content": (
                                "You are a professional simultaneous interpreter for a Zoom meeting. "
                                "Translate the English input to natural, conversational French. "
                                f"Previous conversation context: {context}. "
                                "If the input is a fragment, try to complete the thought based on context. "
                                "Output ONLY the French translation."
                            )
                        },
                        {"role": "user", "content": text}
                    ],
                    temperature=0.0,
                    stream=True 
                )

                self.log(f"[MT] Translating: '{text}'")
                full_translation = ""
                
                for chunk in completion:
                    if chunk.choices[0].delta.content:
                        full_translation += chunk.choices[0].delta.content

                # Put the complete sentence once so TTS gets a full phrase,
                # not a stream of individual tokens that break the WS session.
                if full_translation.strip():
                    self.output_queue.put(full_translation.strip())

                self.context_history.append(text)
                if len(self.context_history) > self.max_context:
                    self.context_history.pop(0)

                duration = time.time() - start_time
                self.log(f"[MT] Result ({duration:.2f}s): {full_translation.strip()}")

            except Exception as e:
                self.log(f"[MT] API Error: {e}")
            
            self.input_queue.task_done()

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 1.0) -> None:
        self.is_running = False
        # Wake thread if blocked on input_queue.get()
        try:
            self.input_queue.put_nowait("")
        except Exception:
            pass

        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None