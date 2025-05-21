import time
import json
import psutil
from llama_cpp import Llama


class TextSummarizer:
    def __init__(self, config_path="config.json", text_path="transcript.txt"):
        self.cpu_threads = psutil.cpu_count(logical=True)
        self.transcript = text_path

        # Загружаем конфигурацию
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        model_path = config.get("saiga_model_path")
        if not model_path:
            raise ValueError("В конфиге не найден ключ 'saiga_model_path'")

        self.model = Llama(
            model_path=model_path,
            n_ctx=8192,
            rope_scaling={"type": "linear", "factor": 2.0},
            n_threads=self.cpu_threads
        )

    def _build_prompt(self, text: str, mode: str) -> str:
        system = (
            "<|start_header_id|>system<|end_header_id|>\n"
            "Ты — Сайга, русскоязычный автоматический ассистент. Ты разговариваешь с людьми и помогаешь им.\n"
            "<|eot_id|><|start_header_id|>user<|end_header_id|>\n"
        )

        user_prompts = {
            "markdown": (
                "Составь подробный, логичный и понятный конспект следующего текста в формате markdown. "
                "Убери слова-паразиты, оставь суть. Англицизмы и формулы допустимы:\n\n"
            ),
            "latex": (
                "Составь подробный, логичный и понятный конспект следующего текста в формате latex. "
                "Убери слова-паразиты, оставь суть. Англицизмы и формулы допустимы:\n\n"
            ),
            "txt": (
                "Составь подробный, логичный и понятный конспект следующего текста. "
                "Убери слова-паразиты, оставь суть. Англицизмы и формулы допустимы:\n\n"
            ),
            "note": (
                "Оформи текст с правильной орфографией и пунктуацией, замени неподходящие слова, "
                "убери ненормативную лексику, сохрани суть:\n\n"
            )
        }

        return system + user_prompts.get(mode, user_prompts["txt"]) + text + "\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"

    def load_transcript(self) -> str:
        with open(self.transcript, "r", encoding="utf-8") as f:
            return f.read().replace("\n", " ").strip()

    def summarize(self, text: str, mode: str = "txt", max_tokens: int = 1024) -> str:
        prompt = self._build_prompt(text, mode)
        start = time.perf_counter()
        output = self.model(prompt=prompt, max_tokens=max_tokens, stop=["</s>"])
        end = time.perf_counter()
        print(f"Время генерации: {end - start:.2f} сек")
        return output["choices"][0]["text"].strip()

    def summarize_from_file(self, mode: str = "txt", max_tokens: int = 1024) -> str:
        text = self.load_transcript()
        return self.summarize(text, mode=mode, max_tokens=max_tokens)
