import time
import requests
import os
from transformers import AutoTokenizer


class QwenAPI():
    def __init__(self, url, tokenizer_path=None, model_name="Qwen"):
        self.url = url
        self.model_name = model_name
        self.guided_decoding_backend = os.environ.get(
            "STRUCTRAG_GUIDED_DECODING_BACKEND", "lm-format-enforcer")

        print("loading tokenizer")
        resolved_tokenizer_path = None
        for candidate in [
            tokenizer_path,
            os.environ.get("STRUCTRAG_TOKENIZER_PATH"),
            "/mnt/data/lizhuoqun/hf_models/gpt2",
        ]:
            if candidate and os.path.exists(candidate):
                resolved_tokenizer_path = candidate
                break

        if resolved_tokenizer_path is None:
            raise Exception("No tokenizer path found. Please pass --tokenizer_path or set STRUCTRAG_TOKENIZER_PATH.")

        self.tokenizer = AutoTokenizer.from_pretrained(resolved_tokenizer_path, trust_remote_code=True)
        print("loading tokenizer done")

    def response(self, input_text, max_new_tokens=4096):
        current_time = time.time()
        
        input_text_len = len(self.tokenizer(input_text)['input_ids'])
        print(f"input_text_len: {input_text_len}")
        if input_text_len > 128000:
            print(f"input_text_len: {input_text_len}", "we reduce the input_text_len")
            input_text = input_text[:int(len(input_text)*(128000/input_text_len))]

        url = self.url
        headers = {
            "Authorization": "EMPTY",
            "Content-Type": "application/json",
        }
        raw_info = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": input_text}],
            "seed": 1024,
            "max_tokens": max_new_tokens,
            "guided_decoding_backend": self.guided_decoding_backend,
        }

        try_time = 0
        response = None
        while try_time < 3:
            try_time += 1

            try:
                callback = requests.post(url, headers=headers, json=raw_info, timeout=(10000, 10000))
                print("callback.status_code", callback.status_code)
            except Exception as e:
                print(f"(print in qwenapi.py callback, try_time {try_time}) Error: {e}")
                continue

            try:
                result = callback.json()
            except Exception as e:
                print(f"(print in qwenapi.py json, try_time {try_time}) status={callback.status_code} text={callback.text[:500]} Error: {e}")
                continue

            usage = result.get("usage", {})
            if usage:
                print(f"prompt_tokens: {usage.get('prompt_tokens')}, total_tokens: {usage.get('total_tokens')}, completion_tokens: {usage.get('completion_tokens')}")

            if callback.status_code != 200:
                error_message = result.get("message")
                if error_message is None and isinstance(result.get("error"), dict):
                    error_message = result["error"].get("message")
                if error_message is None:
                    error_message = str(result)
                print(f"(print in qwenapi.py response, try_time {try_time}) callback: {result}")
                if "Please reduce the length of the messages" in error_message:
                    current_tokne_len = error_message.split("However, you requested")[1].split("tokens in the messages, Please")[0].strip()
                    current_tokne_len = int(current_tokne_len)
                    print(f"current_tokne_len: {current_tokne_len}")
                    raw_info = {
                        "model": self.model_name,
                        "messages": [{"role": "user", "content": input_text[:int(len(input_text)*(128000/current_tokne_len))]}],
                        "seed": 1024,
                        "max_tokens": max_new_tokens,
                        "guided_decoding_backend": self.guided_decoding_backend,
                    }
                continue

            try:
                response = result['choices'][0]['message']['content']
                break
            except Exception as e:
                print(f"(print in qwenapi.py parse, try_time {try_time}) callback: {result} Error: {e}")
                continue

        if response is None:
            raise Exception(f"response is None")

        print("used time in this qwenapi:", (time.time()-current_time)/60, "min")
        return response
