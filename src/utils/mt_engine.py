import os, queue, threading, time
from groq import Groq

class MTProcessor:
    def __init__(self, input_queue, output_queue, log_callback):
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.log = log_callback
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model_id = "llama-3.3-70b-versatile"
        
        # context buffer for gendered pronoun resolution and handling fragmented sentences
        self.context_history = [] 
        self.max_context = 3

    def _get_context_string(self):
        return " ".join(self.context_history)

    def _run(self):
        self.log("[MT] Groq API Translation Engine Online")
        while True:
            text = self.input_queue.get()
            if not text.strip(): continue

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
                        token = chunk.choices[0].delta.content
                        full_translation += token
                        self.output_queue.put(token)
                self.context_history.append(text)
                if len(self.context_history) > self.max_context:
                    self.context_history.pop(0)

                duration = time.time() - start_time
                self.log(f"[MT] Result ({duration:.2f}s): {full_translation.strip()}")

            except Exception as e:
                self.log(f"[MT] API Error: {e}")
            
            self.input_queue.task_done()

    def start(self):
        threading.Thread(target=self._run, daemon=True).start()