from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional


class LocalChatModel:
    _cache: dict[tuple[str, str, bool], tuple[Any, Any, Any]] = {}

    def __init__(
        self,
        *,
        model_dir: str | Path,
        compute_dtype: str = "bfloat16",
        enable_thinking: Optional[bool] = False,
        max_input_tokens: int = 32768,
        trust_remote_code: bool = True,
    ) -> None:
        self.model_dir = str(Path(model_dir).expanduser().resolve())
        self.compute_dtype = compute_dtype
        self.enable_thinking = enable_thinking
        self.max_input_tokens = max_input_tokens
        self.trust_remote_code = trust_remote_code

        key = (self.model_dir, self.compute_dtype, self.trust_remote_code)
        cached = self._cache.get(key)
        if cached is None:
            tokenizer, model, device = self._load_model()
            self._cache[key] = (tokenizer, model, device)
        else:
            tokenizer, model, device = cached

        self.tokenizer = tokenizer
        self.model = model
        self.device = device

    def _load_model(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
            "auto": "auto",
        }
        torch_dtype = dtype_map.get(str(self.compute_dtype).strip().lower(), "auto")

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_dir,
            trust_remote_code=self.trust_remote_code,
        )
        if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
            tokenizer.pad_token = tokenizer.eos_token

        model_kwargs: Dict[str, Any] = {
            "device_map": "auto",
            "trust_remote_code": self.trust_remote_code,
            "low_cpu_mem_usage": True,
        }
        if torch_dtype != "auto":
            model_kwargs["torch_dtype"] = torch_dtype

        model = AutoModelForCausalLM.from_pretrained(
            self.model_dir,
            **model_kwargs,
        )
        model.eval()

        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        return tokenizer, model, device

    def _build_prompt(self, system_prompt: str, user_prompt: str) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        chat_kwargs: Dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if self.enable_thinking is not None:
            chat_kwargs["chat_template_kwargs"] = {"enable_thinking": self.enable_thinking}

        try:
            return self.tokenizer.apply_chat_template(messages, **chat_kwargs)
        except TypeError:
            chat_kwargs.pop("chat_template_kwargs", None)
            return self.tokenizer.apply_chat_template(messages, **chat_kwargs)

    def generate_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_output_tokens: int = 400,
        temperature: float = 0.0,
    ) -> str:
        import torch

        prompt_text = self._build_prompt(system_prompt, user_prompt)
        inputs = self.tokenizer(prompt_text, return_tensors="pt")
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")

        if input_ids.shape[-1] > self.max_input_tokens:
            input_ids = input_ids[:, -self.max_input_tokens :]
            if attention_mask is not None:
                attention_mask = attention_mask[:, -self.max_input_tokens :]

        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        do_sample = float(temperature or 0.0) > 0.0
        generation_kwargs: Dict[str, Any] = {
            "max_new_tokens": int(max_output_tokens),
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            generation_kwargs["temperature"] = float(temperature)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **generation_kwargs,
            )

        new_tokens = output_ids[0, input_ids.shape[-1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
