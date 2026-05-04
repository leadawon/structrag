import time
import requests
import os
import re
from transformers import AutoTokenizer


class QwenAPI():
    def __init__(self, url, tokenizer_path=None, model_name="Qwen", guided_decoding_backend="__auto__", trace_logger=None):
        self.url = url
        self.model_name = model_name
        self.trace_logger = trace_logger
        if guided_decoding_backend == "__auto__":
            self.guided_decoding_backend = os.environ.get(
                "STRUCTRAG_GUIDED_DECODING_BACKEND", "lm-format-enforcer")
        else:
            self.guided_decoding_backend = guided_decoding_backend
        self.max_input_tokens = self._parse_optional_int(
            os.environ.get("STRUCTRAG_MAX_INPUT_TOKENS"))
        self.default_max_new_tokens = self._parse_optional_int(
            os.environ.get("STRUCTRAG_MAX_NEW_TOKENS")) or 4096
        self.enable_thinking = self._parse_optional_bool(
            os.environ.get("STRUCTRAG_ENABLE_THINKING"))
        self.merge_reasoning = self._parse_optional_bool(
            os.environ.get("STRUCTRAG_MERGE_REASONING")) or False

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

    def _parse_optional_int(self, raw_value):
        if raw_value is None or raw_value == "":
            return None
        return int(raw_value)

    def _parse_optional_bool(self, raw_value):
        if raw_value is None or raw_value == "":
            return None
        return str(raw_value).strip().lower() in {"1", "true", "yes", "on"}

    def _token_len(self, text):
        return len(self.tokenizer(text)["input_ids"])

    def _build_request_payload(self, input_text, max_new_tokens):
        raw_info = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": input_text}],
            "seed": 1024,
            "max_tokens": max_new_tokens,
        }
        if self.guided_decoding_backend:
            raw_info["guided_decoding_backend"] = self.guided_decoding_backend
        if self.enable_thinking is not None:
            raw_info["chat_template_kwargs"] = {
                "enable_thinking": self.enable_thinking,
            }
        return raw_info

    def _extract_message_text(self, result):
        choice = result["choices"][0]
        message = choice.get("message") or {}
        content = message.get("content")
        reasoning = (
            message.get("reasoning")
            or message.get("reasoning_content")
            or ""
        )

        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if text:
                        parts.append(text)
                elif item:
                    parts.append(str(item))
            content = "\n".join(parts)

        if content is None:
            content = ""

        if reasoning and self.merge_reasoning:
            return f"<think>{reasoning}</think>\n{content}", "content_with_reasoning"
        if content:
            return content, "content"
        if reasoning:
            return reasoning, "reasoning_fallback"
        return "", "empty"

    def _truncate_to_token_limit(self, text, token_limit):
        token_ids = self.tokenizer(text)["input_ids"]
        if len(token_ids) <= token_limit:
            return text, len(token_ids), False
        truncated_ids = token_ids[:token_limit]
        truncated_text = self.tokenizer.decode(
            truncated_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        return truncated_text, len(truncated_ids), True

    def _parse_context_limit_error(self, error_message):
        max_match = re.search(r"maximum context length is (\d+) tokens", error_message)
        requested_match = re.search(r"However, you requested (\d+) tokens", error_message)
        prompt_match = re.search(r"\((\d+) in the messages", error_message)
        completion_match = re.search(r"(\d+) in the completion\)", error_message)
        max_tokens = int(max_match.group(1)) if max_match else None
        requested_tokens = int(requested_match.group(1)) if requested_match else None
        prompt_tokens = int(prompt_match.group(1)) if prompt_match else None
        completion_tokens = int(completion_match.group(1)) if completion_match else None
        return max_tokens, requested_tokens, prompt_tokens, completion_tokens

    def response(self, input_text, max_new_tokens=None, trace_context=None):
        trace_context = trace_context or {}
        if max_new_tokens is None:
            max_new_tokens = self.default_max_new_tokens
        current_time = time.time()
        truncated = False
        if self.max_input_tokens is not None:
            input_text, input_text_len, truncated = self._truncate_to_token_limit(
                input_text, self.max_input_tokens)
            if truncated:
                print(f"input_text_len exceeds configured limit. truncate_to={self.max_input_tokens}")
        else:
            input_text_len = self._token_len(input_text)
        print(f"input_text_len: {input_text_len}")

        url = self.url
        headers = {
            "Authorization": "EMPTY",
            "Content-Type": "application/json",
        }
        raw_info = self._build_request_payload(input_text, max_new_tokens)

        try_time = 0
        response = None
        last_error_message = None
        attempt_errors = []
        usage = {}
        callback_status_code = None
        while try_time < 3:
            try_time += 1

            try:
                callback = requests.post(url, headers=headers, json=raw_info, timeout=(10000, 10000))
                print("callback.status_code", callback.status_code)
                callback_status_code = callback.status_code
            except Exception as e:
                last_error_message = str(e)
                attempt_errors.append({"attempt": try_time, "error": str(e)})
                print(f"(print in qwenapi.py callback, try_time {try_time}) Error: {e}")
                continue

            result = None
            response_text = callback.text[:500]
            if callback.status_code != 200:
                try:
                    result = callback.json()
                except Exception:
                    result = None

                error_message = None
                if isinstance(result, dict):
                    error_message = result.get("message")
                    if error_message is None and isinstance(result.get("error"), dict):
                        error_message = result["error"].get("message")
                    if error_message is None:
                        error_message = str(result)
                if error_message is None:
                    error_message = response_text

                last_error_message = f"status={callback.status_code} text={error_message}"
                attempt_errors.append(
                    {
                        "attempt": try_time,
                        "status_code": callback.status_code,
                        "error": error_message,
                    }
                )
                print(
                    f"(print in qwenapi.py response, try_time {try_time}) "
                    f"status={callback.status_code} text={error_message}"
                )
                if "Please reduce the length of the messages" in error_message:
                    max_context_tokens, requested_tokens, prompt_tokens, completion_tokens = self._parse_context_limit_error(error_message)
                    target_limit = self.max_input_tokens
                    effective_max_new_tokens = max_new_tokens
                    if max_context_tokens is not None:
                        reserve_for_completion = completion_tokens if completion_tokens is not None else max_new_tokens
                        # Keep a small headroom for chat template/system tokens.
                        safe_limit = max(1, max_context_tokens - reserve_for_completion - 256)
                        target_limit = safe_limit if target_limit is None else min(target_limit, safe_limit)
                        self.max_input_tokens = target_limit
                    if max_context_tokens is not None and prompt_tokens is not None:
                        allowed_completion = max(32, max_context_tokens - prompt_tokens - 128)
                        effective_max_new_tokens = min(max_new_tokens, allowed_completion)
                    if target_limit is not None:
                        input_text, truncated_len, truncated = self._truncate_to_token_limit(
                            input_text, target_limit)
                        print(
                            f"reduce messages length: requested_tokens={requested_tokens}, "
                            f"target_limit={target_limit}, truncated_tokens={truncated_len}, "
                            f"truncated={truncated}, max_new_tokens={effective_max_new_tokens}"
                        )
                        raw_info = self._build_request_payload(
                            input_text,
                            effective_max_new_tokens,
                        )
                continue

            try:
                result = callback.json()
            except Exception as e:
                last_error_message = f"status={callback.status_code} text={response_text} parse_error={e}"
                print(f"(print in qwenapi.py json, try_time {try_time}) status={callback.status_code} text={response_text} Error: {e}")
                continue

            usage = result.get("usage", {})
            if usage:
                print(f"prompt_tokens: {usage.get('prompt_tokens')}, total_tokens: {usage.get('total_tokens')}, completion_tokens: {usage.get('completion_tokens')}")

            try:
                response, response_source = self._extract_message_text(result)
                if response_source != "content":
                    print(f"response_source: {response_source}")
                break
            except Exception as e:
                last_error_message = f"response_parse_error={e}; payload={result}"
                attempt_errors.append({"attempt": try_time, "error": last_error_message})
                print(f"(print in qwenapi.py parse, try_time {try_time}) callback: {result} Error: {e}")
                continue

        data_id = trace_context.get("data_id")
        component = trace_context.get("component", "llm.response")
        metadata = dict(trace_context.get("metadata") or {})
        metadata.update(
            {
                "url": self.url,
                "model_name": self.model_name,
                "guided_decoding_backend": self.guided_decoding_backend,
                "enable_thinking": self.enable_thinking,
                "merge_reasoning": self.merge_reasoning,
                "max_tokens": max_new_tokens,
                "input_text_len": input_text_len,
                "truncated_input": truncated,
                "attempts": try_time,
                "status_code": callback_status_code,
                "usage": usage,
            }
        )
        if attempt_errors:
            metadata["attempt_errors"] = attempt_errors

        if response is None:
            if self.trace_logger and data_id is not None:
                self.trace_logger.log_llm_call(
                    data_id=data_id,
                    component=component,
                    prompt=input_text,
                    response=f"ERROR: {last_error_message}",
                    metadata=metadata,
                )
            raise Exception(f"response is None; last_error={last_error_message}")

        if self.trace_logger and data_id is not None:
            self.trace_logger.log_llm_call(
                data_id=data_id,
                component=component,
                prompt=input_text,
                response=response,
                metadata=metadata,
            )

        print("used time in this qwenapi:", (time.time()-current_time)/60, "min")
        return response
