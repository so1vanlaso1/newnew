"""vLLM-based translator inference with optional LoRA adapter.

This module owns the heavyweight LLM handle. Importing it triggers a vLLM
load (slow), so the pipeline keeps a single shared instance via
`get_translator()`. Tests can stub out `LLMBackend` directly.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol

from data.types import AnswerType, Record, Translation
from translator.parse import extract_goal_expr, parse_translator_output
from translator.prompt import build_messages_for_record
from translator.repair import repair_program

log = logging.getLogger(__name__)

# A PEFT adapter weight is keyed `base_model.model.<base-module-path>.lora_A.weight`.
# We recover `<base-module-path>` to check it exists in the base model.
_ADAPTER_KEY = re.compile(r"^base_model\.model\.(.+?)\.lora_[AB]\b")


def _adapter_target_paths(adapter_keys: Iterable[str]) -> set[str]:
    """Base-model module paths a LoRA adapter expects to wrap."""
    out: set[str] = set()
    for k in adapter_keys:
        m = _ADAPTER_KEY.match(k)
        if m:
            out.add(m.group(1))
    return out


def _read_adapter_keys(lora_path: str) -> list[str]:
    p = Path(lora_path)
    st = p / "adapter_model.safetensors"
    if st.exists():
        from safetensors import safe_open  # type: ignore[import-not-found]

        with safe_open(str(st), framework="pt") as f:
            return list(f.keys())
    binf = p / "adapter_model.bin"
    if binf.exists():
        import torch  # type: ignore[import-not-found]

        return list(torch.load(binf, map_location="cpu").keys())
    return []


def assert_adapter_matches_base(base_model: object, lora_path: str) -> None:
    """Fail loudly if a LoRA adapter targets modules the base model does not have.

    This catches the silent failure where an adapter trained on one model (e.g.
    a dense text Qwen with `model.layers.*` self-attention) is loaded onto a
    differently-structured base (e.g. a multimodal model with text layers under
    `model.language_model.layers.*` and linear-attention blocks). PEFT would
    apply nothing, leaving you running the *bare base model* — which looks like a
    weak/garbage fine-tune. Better to stop with an actionable message."""
    expected = _adapter_target_paths(_read_adapter_keys(lora_path))
    if not expected:
        return  # can't introspect the adapter; don't block
    base_modules = {name for name, _ in base_model.named_modules()}  # type: ignore[attr-defined]
    hits = expected & base_modules
    if not hits:
        sample_exp = sorted(expected)[:3]
        sample_base = sorted(
            m for m in base_modules if m.endswith(("q_proj", "down_proj"))
        )[:3] or sorted(base_modules)[:3]
        raise RuntimeError(
            "LoRA/base mismatch: none of the adapter's "
            f"{len(expected)} target modules exist in the loaded base model.\n"
            f"  adapter expects : {sample_exp}\n"
            f"  base exposes    : {sample_base}\n"
            f"  adapter dir     : {lora_path}\n"
            "The adapter was fine-tuned on a DIFFERENT model than this base. Point "
            "--model at the same base it was trained on (its adapter_config.json "
            "records `base_model_name_or_path`), or pass --no-lora to run the base "
            "model alone."
        )
    if len(hits) < len(expected):
        log.warning(
            "LoRA/base partial match: %d/%d adapter target modules found in base; "
            "the rest will not apply.",
            len(hits), len(expected),
        )


# Qwen's ChatML template. A *base* (non-instruct) model may ship no chat
# template at all; we install this so apply_chat_template() behaves identically
# at training and inference time. Harmless when the model already has one.
QWEN_CHATML_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)


@dataclass
class TranslatorConfig:
    # Single workhorse for the whole pipeline: Qwen/Qwen3.5-4B-Base
    # (https://huggingface.co/Qwen/Qwen3.5-4B-Base). Defaults target an
    # RTX 5070 (12 GB GDDR7, Blackwell sm_120). The base model ships bf16
    # weights, so vLLM loads it UNQUANTIZED by default (quantization=None) — do
    # not set awq/gptq here, the base model has no pre-quantized checkpoint. bf16
    # (~8 GB) is tight on 12 GB; use fp8 or the transformers 4-bit path if it OOMs.
    model: str = "Qwen/Qwen3.5-4B-Base"
    quantization: str | None = None           # None=bf16 | fp8 (lighter on 12 GB) | bitsandbytes | awq_marlin
    load_format: str | None = None            # set "bitsandbytes" for vLLM in-flight 4-bit
    dtype: str = "bfloat16"                    # Blackwell-native; "auto" also resolves to bf16
    max_model_len: int = 4096                 # premises are short; lower to 3072 if KV cache is tight
    gpu_memory_utilization: float = 0.80      # leave ~2.4 GB free for Z3 + Python on 12 GB
    enable_lora: bool = True
    max_lora_rank: int = 32
    lora_path: str | None = None              # set after fine-tune
    k_samples: int = 5
    temperature: float = 0.3
    top_p: float = 0.9
    max_new_tokens: int = 1024
    n_fewshot: int = 2


class LLMBackend(Protocol):
    def chat_generate(
        self,
        batch_messages: list[list[dict]],
        n: int,
        temperature: float,
        top_p: float,
        max_tokens: int,
        lora_path: str | None,
    ) -> list[list[str]]:
        """Return per-prompt, per-sample raw completion strings."""
        ...


class VLLMBackend:
    def __init__(self, cfg: TranslatorConfig):
        # Imports are lazy so the rest of the package can be imported on a box
        # without vLLM installed (Windows dev, tests, etc.).
        from vllm import LLM, SamplingParams  # type: ignore[import-not-found]

        self._SamplingParams = SamplingParams
        llm_kwargs = dict(
            model=cfg.model,
            dtype=cfg.dtype,
            max_model_len=cfg.max_model_len,
            gpu_memory_utilization=cfg.gpu_memory_utilization,
            enable_lora=cfg.enable_lora,
            max_lora_rank=cfg.max_lora_rank,
            trust_remote_code=True,
        )
        # Only pass a quantization scheme when one is explicitly requested. The
        # base model is plain bf16; passing awq/gptq would make vLLM hunt for a
        # pre-quantized checkpoint that doesn't exist and crash on load.
        if cfg.quantization:
            llm_kwargs["quantization"] = cfg.quantization
        if cfg.load_format:
            llm_kwargs["load_format"] = cfg.load_format
        self.llm = LLM(**llm_kwargs)
        self.tokenizer = self.llm.get_tokenizer()
        if self.tokenizer.chat_template is None:
            self.tokenizer.chat_template = QWEN_CHATML_TEMPLATE
            log.warning("tokenizer has no chat_template; installed Qwen ChatML fallback")

    def chat_generate(
        self,
        batch_messages: list[list[dict]],
        n: int,
        temperature: float,
        top_p: float,
        max_tokens: int,
        lora_path: str | None,
    ) -> list[list[str]]:
        prompts = [
            self.tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in batch_messages
        ]
        sp = self._SamplingParams(
            n=n, temperature=temperature, top_p=top_p, max_tokens=max_tokens
        )
        lora_request = None
        if lora_path:
            from vllm.lora.request import LoRARequest  # type: ignore[import-not-found]

            lora_request = LoRARequest("translator", 1, lora_path)
        outputs = self.llm.generate(prompts, sampling_params=sp, lora_request=lora_request)
        return [[c.text for c in out.outputs] for out in outputs]


class TransformersBackend:
    """HuggingFace transformers + PEFT backend.

    Slower than vLLM but dependency-light and runs anywhere PyTorch runs, which
    makes it the convenient path for loading a *local* base model directory plus
    a *local* LoRA adapter directory (no HF-hub download, no vLLM build). The
    base weights are loaded once; the LoRA adapter is loaded once and toggled
    per call: a falsy `lora_path` disables the adapter so the CoT fallback runs
    on the base model, exactly like the vLLM path.
    """

    def __init__(
        self,
        cfg: TranslatorConfig,
        load_4bit: bool = False,
        device_map: str = "auto",
        enable_thinking: bool = True,
        stop_strings: list[str] | None = None,
        dtype: str = "bfloat16",
    ):
        # `load_4bit` vs `dtype` is the inference-precision switch:
        #   load_4bit=True  → NF4 4-bit base weights (bnb), ~2.5 GB, bf16 compute.
        #   load_4bit=False → full `dtype` weights (default bf16), ~8 GB on a 4B.
        # On the 5070's 12 GB, 4-bit is the comfortable choice; bf16 fits but is tight.
        # `enable_thinking` is forwarded to the Qwen chat template: True opens a
        # `<think>` block (the model reasons before answering); False pre-fills an
        # empty, already-closed think block so it emits the answer directly.
        self.enable_thinking = enable_thinking
        # `stop_strings` halts generation as soon as any of these substrings is
        # produced. Setting it to ["</z3py>"] for translation stops the model the
        # instant it closes the program block, instead of letting it ramble on
        # (hallucinating fresh fake turns) until max_new_tokens — a large CPU
        # speedup and cleaner output. Left None for the CoT path (which stops on
        # its own "FINAL ANSWER" convention via max_new_tokens).
        self.stop_strings = stop_strings
        # Lazy imports: keep the package importable on boxes without torch.
        import torch  # type: ignore[import-not-found]
        from transformers import (  # type: ignore[import-not-found]
            AutoModelForCausalLM,
            AutoTokenizer,
        )

        self._torch = torch
        torch_dtype = getattr(torch, dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.tokenizer.chat_template is None:
            self.tokenizer.chat_template = QWEN_CHATML_TEMPLATE
            log.warning("tokenizer has no chat_template; installed Qwen ChatML fallback")

        quant_config = None
        if load_4bit:
            from transformers import BitsAndBytesConfig  # type: ignore[import-not-found]

            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,  # bf16 compute on Blackwell
                bnb_4bit_use_double_quant=True,
            )

        model = AutoModelForCausalLM.from_pretrained(
            cfg.model,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            device_map=device_map,
            quantization_config=quant_config,
        )

        self._has_adapter = False
        if cfg.enable_lora and cfg.lora_path:
            from peft import PeftModel  # type: ignore[import-not-found]

            # Preflight: refuse to silently run the bare base model when the
            # adapter targets modules this base doesn't have.
            assert_adapter_matches_base(model, cfg.lora_path)
            model = PeftModel.from_pretrained(model, cfg.lora_path)
            self._has_adapter = True

        model.eval()
        self.model = model

    def render_prompt(self, messages: list[dict]) -> str:
        """Render the exact chat-template text passed to the model."""
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )

    def chat_generate(
        self,
        batch_messages: list[list[dict]],
        n: int,
        temperature: float,
        top_p: float,
        max_tokens: int,
        lora_path: str | None,
    ) -> list[list[str]]:
        import contextlib

        torch = self._torch
        # Adapter is active unless this call explicitly asked for the base model
        # (lora_path falsy) — that is how the CoT fallback gets the base weights.
        want_adapter = bool(lora_path) and self._has_adapter
        toggle: contextlib.AbstractContextManager = contextlib.nullcontext()
        if self._has_adapter and not want_adapter:
            toggle = self.model.disable_adapter()

        do_sample = temperature and temperature > 0.0
        results: list[list[str]] = []
        with toggle, torch.no_grad():
            for messages in batch_messages:
                prompt = self.render_prompt(messages)
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
                gen_kwargs = dict(
                    do_sample=bool(do_sample),
                    temperature=max(temperature, 1e-5) if do_sample else None,
                    top_p=top_p if do_sample else None,
                    num_return_sequences=n,
                    max_new_tokens=max_tokens,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
                if self.stop_strings:
                    # stop_strings requires the tokenizer to be passed to generate().
                    gen_kwargs["stop_strings"] = self.stop_strings
                    gen_kwargs["tokenizer"] = self.tokenizer
                gen = self.model.generate(**inputs, **gen_kwargs)
                prompt_len = inputs["input_ids"].shape[1]
                completions = [
                    self.tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)
                    for seq in gen
                ]
                results.append(completions)
        return results


class Translator:
    def __init__(self, backend: LLMBackend, cfg: TranslatorConfig):
        self.backend = backend
        self.cfg = cfg

    def translate(self, record: Record) -> list[list[Translation]]:
        """Return per-prompt list of K Translation candidates.

        - Yes/No/Uncertain → outer list has 1 entry, inner list has K.
        - MCQ → outer list has len(options) entries, inner list has K each.
        - Open-ended → empty outer list.
        """
        if record.answer_type == AnswerType.OPEN_ENDED:
            return []

        batches = build_messages_for_record(record, n_fewshot=self.cfg.n_fewshot)
        if not batches:
            return []

        raw = self.backend.chat_generate(
            batch_messages=batches,
            n=self.cfg.k_samples,
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            max_tokens=self.cfg.max_new_tokens,
            lora_path=self.cfg.lora_path,
        )

        translations: list[list[Translation]] = []
        for prompt_outputs in raw:
            per_prompt: list[Translation] = []
            for i, txt in enumerate(prompt_outputs):
                code = parse_translator_output(txt)
                if code is None:
                    continue
                # Auto-declare any symbol the model used but forgot to declare,
                # so an otherwise-runnable program is not lost to a NameError.
                code = repair_program(code)
                per_prompt.append(
                    Translation(
                        code=code,
                        goal_expr=extract_goal_expr(code),
                        raw_text=txt,
                        sample_index=i,
                    )
                )
            translations.append(per_prompt)
        return translations


_GLOBAL_TRANSLATOR: Translator | None = None


def get_translator(cfg: TranslatorConfig | None = None) -> Translator:
    """Lazy singleton — one vLLM load per process."""
    global _GLOBAL_TRANSLATOR
    if _GLOBAL_TRANSLATOR is None:
        cfg = cfg or TranslatorConfig()
        backend = VLLMBackend(cfg)
        _GLOBAL_TRANSLATOR = Translator(backend, cfg)
    return _GLOBAL_TRANSLATOR
