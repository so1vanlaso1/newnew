"""The single translator: fvossel/Llama-3.1-8B-Instruct-nl-to-fol (NL → FOL) + the
existing FOL→Z3 bridge.

This is the ONE model the whole pipeline runs on. It is a PEFT/LoRA adapter
(https://huggingface.co/fvossel/Llama-3.1-8B-Instruct-nl-to-fol) on top of the
gated base ``meta-llama/Llama-3.1-8B-Instruct``. We load the base in 4-bit (NF4,
bitsandbytes) and apply the adapter, then drive it in three roles from ONE
resident model:

    role 1  NL → FOL          (adapter ON)   — per-sentence translation
    role 2  predicate grouping (adapter OFF)  — base chat clusters synonym names
    role 3  CoT fallback       (adapter OFF)  — base chat reasons when symbolic fails

The pipeline, solver, voting, predicate-grouping, FOL-repair, and explanation
stages are reused unchanged — this module only produces the Z3 program:

    NL premise/option ─► Llama+adapter ─► one FOL formula (Unicode: ∀x (P(x) → Q(x)))
                                          │
    all premise FOLs + goal FOL ─────────┴─► fol_converter.convert_premises_to_z3py
                                          └─► assemble `premises = [...]` + `goal = ...`
                                          └─► Z3-Python program  (Translation.code)

The fvossel adapter was trained to translate ONE declarative sentence at a time
and to answer with ``𝜙=`` followed by a single FOL formula using the operators
∀ ∃ ¬ ∧ ∨ → ↔ ⊕ (see the model card / SYSTEM_PROMPT below). ``_strip_phi`` peels
the ``𝜙=`` marker; ``normalize_fol`` + ``fol_converter`` handle the rest.

Because the *base* model is a capable Llama-3.1-8B-Instruct chat model, disabling
the adapter (``model.disable_adapter()``) gives a general chat backend for the
grouping and CoT stages — so the full pipeline runs on this single model, with no
second model load.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol

from data.types import AnswerType, Record, Translation
from translator.fol_converter import convert_premises_to_z3py
from translator.fol_repair import add_type_facts, ground_goal

log = logging.getLogger(__name__)

# The exact system prompt the fvossel adapter was fine-tuned with (model card).
# It instructs single-formula output prefixed with '𝜙='.
SYSTEM_PROMPT = (
    "You are a helpful AI assistant that translates Natural Language (NL) text "
    "into First-Order Logic (FOL) using only the given quantors and junctors: "
    "∀ (for all), ∃ (there exists), ¬ (not), ∧ (and), ∨ (or), → (implies), "
    "↔ (if and only if), ⊕ (xor). "
    "Start your answer with '𝜙=' followed by the FOL-formula. Do not include any other text."
)


# ─────────────────────────────────────────────────────────────────────────
# PEFT adapter / base-model compatibility check (moved here from the old
# translator.infer; the only consumer now is the Llama backend + its tests).
# ─────────────────────────────────────────────────────────────────────────

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
    """Read adapter weight key names from a LOCAL adapter dir. Returns [] for a
    HF repo id (nothing on disk yet) — the check then no-ops, which is correct:
    PeftModel.from_pretrained will download + validate the repo itself."""
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
    """Fail loudly if a LOCAL LoRA adapter targets modules the base model lacks.

    Catches the silent failure where an adapter trained on a differently-structured
    base applies nothing, leaving you running the bare base model. No-ops for a HF
    repo id (no local weights to introspect)."""
    expected = _adapter_target_paths(_read_adapter_keys(lora_path))
    if not expected:
        return  # repo id, or can't introspect — don't block
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
            "Point --base-model at the same base the adapter was trained on "
            "(its adapter_config.json records `base_model_name_or_path`)."
        )
    if len(hits) < len(expected):
        log.warning(
            "LoRA/base partial match: %d/%d adapter target modules found in base; "
            "the rest will not apply.",
            len(hits), len(expected),
        )


# ─────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────


@dataclass
class LlamaFolConfig:
    # The fvossel adapter is a LoRA on the GATED meta-llama/Llama-3.1-8B-Instruct.
    base_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    adapter: str = "fvossel/Llama-3.1-8B-Instruct-nl-to-fol"
    # 4bit (NF4, bitsandbytes) ≈ 5.5 GB — the intended footprint (needs ~8 GB
    # VRAM with headroom). bf16/fp16 load the full weights (~16 GB).
    precision: str = "4bit"               # 4bit | bf16 | fp16
    # bnb 4-bit compute dtype (and load dtype for the non-quant path). float16 is
    # universally safe; bfloat16 is fine on Ampere+ (use it if your GPU supports it).
    compute_dtype: str = "float16"        # float16 | bfloat16
    max_input_len: int = 512              # premise/option sentences are short
    max_new_tokens: int = 256             # one FOL formula; caps the decode
    # K=1 greedy/beam is the model's intended deterministic use. K>1 switches to
    # sampling to feed the vote machinery diverse candidates.
    num_beams: int = 1                    # >1 → beam search at K=1
    batch_size: int = 1                   # sentences per generate() call (left-padded)
    k_samples: int = 1
    temperature: float = 0.7
    top_p: float = 0.9
    # When set, every record's NL→FOL trace is appended as JSON lines here.
    dump_io_path: str | None = None
    # Drop a record if more than this fraction of its premises fail to parse.
    max_skip_fraction: float = 0.25
    # ── Deterministic FOL repair bolt-ons (translator.fol_repair) ──────────
    ground_goals: bool = True
    assert_type_facts: bool = True
    type_guard_min_rules: int = 2


# ─────────────────────────────────────────────────────────────────────────
# Model output → FOL surface form
# ─────────────────────────────────────────────────────────────────────────

# The model answers `𝜙=<formula>`. The marker may be the mathematical italic phi
# (U+1D719), a plain Greek phi (U+03C6 / U+03D5), or the literal "phi", possibly
# with surrounding spaces. Peel it (and anything before it) off, then keep the
# first non-empty line — a FOL formula is a single line.
_PHI_MARK = re.compile(r"(?:\U0001D719|φ|ϕ|Φ|phi|Phi|PHI)\s*=\s*")


def _strip_phi(text: str) -> str:
    t = text.replace("\r", "").strip()
    m = _PHI_MARK.search(t)
    if m:
        t = t[m.end():]
    else:
        t2 = t.lstrip()
        if t2.startswith("="):
            t = t2[1:]
    lines = [ln for ln in t.splitlines() if ln.strip()]
    return lines[0].strip() if lines else ""


# fol_converter's tokenizer accepts FORALL/EXISTS / -> <-> ~ / Unicode ∀∃∧∨¬→↔ and
# the doubled ASCII forms. The fvossel model emits Unicode connectives, so this is
# mostly a pass-through; the word-operator mapping below is kept as a safety net for
# any model drift to AND/OR/NOT/IMPLIES/IFF or single &/|.
#
# Word operators are replaced WHOLE-WORD only (\b…\b): a predicate the model glued
# together such as `NOTQualifiesForScholarship` must NOT be mangled.
_WORD_OPS = [
    (re.compile(r"\bIMPLIES\b"), "→"),
    (re.compile(r"\bIFF\b"), "↔"),
    (re.compile(r"\bEQUIV\b"), "↔"),
    (re.compile(r"\bAND\b"), "∧"),
    (re.compile(r"\bOR\b"), "∨"),
    (re.compile(r"\bNOT\b"), "¬"),
]


def normalize_fol(fol: str) -> str:
    s = fol.strip()
    # Drop a trailing period the model sometimes appends after the formula.
    if s.endswith("."):
        s = s[:-1].rstrip()
    # Space out glued quantifier runs ("FORALLxFORALLy") so each tokenizes.
    s = re.sub(r"(FORALL|EXISTS)", r" \1 ", s)
    s = re.sub(r"\s+", " ", s).strip()
    for pattern, symbol in _WORD_OPS:
        s = pattern.sub(symbol, s)
    # Collapse doubled-then-single connectives. Order matters (doubled first).
    s = s.replace("&&", "∧").replace("||", "∨")
    s = s.replace("&", "∧").replace("|", "∨")
    return s


# ─────────────────────────────────────────────────────────────────────────
# FOL list + goal  ─►  Z3-Python program string
# ─────────────────────────────────────────────────────────────────────────


def assemble_z3_program(
    premises_fol: list[str],
    goal_fol: str | None,
    max_skip_fraction: float = 0.25,
    ground_goals: bool = True,
    assert_type_facts: bool = True,
    type_guard_min_rules: int = 2,
) -> tuple[str | None, str | None]:
    """Build a Z3-Python program from FOL strings.

    Returns (code, goal_expr). `code`, when exec'd by solver.z3_runner, populates
    `premises: list[BoolRef]` and `goal: BoolRef`. Returns (None, None) when too
    many premises fail to parse or none survive — the caller then yields no
    translation for the record (it routes to `from_failure`).

    Two deterministic repair passes run before conversion (translator.fol_repair):
      * ground_goals re-grounds a mangled goal to `Pred(entity)`, and
      * assert_type_facts adds free sort-guard facts so gated rules can fire.
    """
    norm_premises = [normalize_fol(p) for p in premises_fol if p and p.strip()]
    if not norm_premises:
        return None, None
    norm_goal = normalize_fol(goal_fol) if goal_fol and goal_fol.strip() else None

    # Bolt-on A: re-ground the goal first (uses the premises to find the entity
    # and the predicate's true arity).
    if ground_goals and norm_goal:
        norm_goal = ground_goal(norm_goal, norm_premises)
    # Bolt-on B: assert free sort guards. Pass the (re-grounded) goal so the
    # goal's own predicate is never asserted as a premise.
    if assert_type_facts:
        norm_premises = add_type_facts(
            norm_premises, norm_goal, min_rules=type_guard_min_rules
        )

    setup, premise_exprs, goal_expr, skipped = convert_premises_to_z3py(
        norm_premises, goal_fol=norm_goal
    )
    # Drop the record if too much of its FOL was unparseable.
    if len(skipped) > max(1, int(len(norm_premises) * max_skip_fraction)):
        return None, None
    if not premise_exprs:
        return None, None

    code = "\n".join(setup)
    code += "\npremises = [\n"
    code += ",\n".join(f"    {p}" for p in premise_exprs)
    code += "\n]\n"
    # A missing/unparseable goal becomes `goal = False`: the goal simply isn't
    # entailed rather than crashing the solve (sound — an open-world predicate).
    code += f"goal = {goal_expr}" if goal_expr is not None else "goal = False"
    return code, goal_expr


# ─────────────────────────────────────────────────────────────────────────
# Backend protocol (for grouping + CoT) — implemented by LlamaFolBackend
# ─────────────────────────────────────────────────────────────────────────


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
        """Per-prompt, per-sample raw completion strings.

        `lora_path` truthy → keep the adapter (NL→FOL); falsy → disable it so the
        base Llama-3.1-8B-Instruct chat model answers (grouping / CoT)."""
        ...


# ─────────────────────────────────────────────────────────────────────────
# The Llama backend: base (4-bit) + fvossel adapter, one resident model
# ─────────────────────────────────────────────────────────────────────────


class LlamaFolBackend:
    """HuggingFace causal-LM backend: base in 4-bit + the fvossel PEFT adapter.

    Two entry points sharing one resident model:
      * ``translate_sentences`` — adapter ON, per-sentence NL→FOL (role 1).
      * ``chat_generate``       — adapter toggled by `lora_path`; falsy disables it
        so the base chat model handles grouping (role 2) and CoT (role 3).
    """

    def __init__(self, cfg: LlamaFolConfig):
        # Lazy imports so the package stays importable on a box without torch.
        import torch  # type: ignore[import-not-found]
        from transformers import (  # type: ignore[import-not-found]
            AutoModelForCausalLM,
            AutoTokenizer,
        )

        self._torch = torch
        self.cfg = cfg
        compute_dtype = getattr(torch, cfg.compute_dtype)

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Left padding is required for correct batched decoding of a decoder-only
        # model, and is what the model card uses.
        self.tokenizer.padding_side = "left"

        has_cuda = bool(torch.cuda.is_available())
        device_map = "auto" if has_cuda else "cpu"

        quant_config = None
        if cfg.precision == "4bit":
            if not has_cuda:
                raise RuntimeError(
                    "precision=4bit needs a CUDA GPU (bitsandbytes). No GPU visible to "
                    "torch — use --precision fp16/bf16 (needs ~16 GB) for a CPU smoke test."
                )
            from transformers import BitsAndBytesConfig  # type: ignore[import-not-found]

            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=compute_dtype,
            )
        elif cfg.precision not in ("bf16", "fp16"):
            raise ValueError(
                f"unknown precision {cfg.precision!r}; use 4bit | bf16 | fp16"
            )

        load_dtype = torch.bfloat16 if cfg.precision == "bf16" else compute_dtype
        if not has_cuda:
            print("[llama] WARNING: no CUDA GPU visible — running on CPU. An 8B model on "
                  "CPU is very slow; use a small --limit.")

        print(f"[llama] loading base {cfg.base_model} (precision={cfg.precision})...")
        model = AutoModelForCausalLM.from_pretrained(
            cfg.base_model,
            trust_remote_code=True,
            torch_dtype=load_dtype,
            device_map=device_map,
            quantization_config=quant_config,
        )

        # Apply the fvossel NL→FOL adapter.
        from peft import PeftModel  # type: ignore[import-not-found]

        assert_adapter_matches_base(model, cfg.adapter)  # no-op for a HF repo id
        print(f"[llama] applying adapter {cfg.adapter}...")
        model = PeftModel.from_pretrained(model, cfg.adapter)
        model.eval()
        self.model = model
        self._has_adapter = True
        self._log_footprint(has_cuda)

    def _log_footprint(self, on_gpu: bool) -> None:
        torch = self._torch
        if on_gpu and torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated(0) / 1e9
            reserved = torch.cuda.memory_reserved(0) / 1e9
            print(f"[llama] loaded precision={self.cfg.precision}: {alloc:.2f} GB allocated, "
                  f"{reserved:.2f} GB reserved on GPU0")
        else:
            print(f"[llama] loaded on CPU (precision={self.cfg.precision})")

    # ── role 1: NL → FOL (adapter ON) ─────────────────────────────────────

    def _render_translate_prompt(self, sentence: str) -> str:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": sentence},
        ]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def translate_sentences(self, sentences: list[str], k: int) -> list[list[str]]:
        """Per-sentence list of k FOL candidate strings (k>=1), adapter active."""
        if not sentences:
            return []
        bs = max(1, self.cfg.batch_size)
        out: list[list[str]] = []
        for start in range(0, len(sentences), bs):
            out.extend(self._translate_chunk(sentences[start : start + bs], k))
        return out

    def _translate_chunk(self, sentences: list[str], k: int) -> list[list[str]]:
        torch = self._torch
        prompts = [self._render_translate_prompt(s) for s in sentences]
        enc = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.cfg.max_input_len,
        ).to(self.model.device)

        gen_kwargs = dict(
            max_new_tokens=self.cfg.max_new_tokens,
            num_return_sequences=k,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        if k > 1:
            gen_kwargs.update(do_sample=True, temperature=self.cfg.temperature, top_p=self.cfg.top_p)
        else:
            gen_kwargs.update(do_sample=False, num_beams=self.cfg.num_beams)

        with torch.no_grad():
            generated = self.model.generate(**enc, **gen_kwargs)
        # Left padding makes every prompt the same width, so slicing off the input
        # length is uniform across the batch.
        in_len = enc["input_ids"].shape[1]
        decoded = self.tokenizer.batch_decode(generated[:, in_len:], skip_special_tokens=True)
        # `decoded` is flat (len(sentences) * k); regroup + strip the 𝜙= marker.
        return [
            [_strip_phi(decoded[i * k + j]) for j in range(k)]
            for i in range(len(sentences))
        ]

    # ── roles 2 & 3: chat (adapter toggled by lora_path) ──────────────────

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
        # Adapter is active unless this call explicitly wants the base model
        # (lora_path falsy) — that is how grouping and CoT get the base weights.
        want_adapter = bool(lora_path) and self._has_adapter
        toggle: contextlib.AbstractContextManager = contextlib.nullcontext()
        if self._has_adapter and not want_adapter:
            toggle = self.model.disable_adapter()

        do_sample = bool(temperature and temperature > 0.0)
        results: list[list[str]] = []
        with toggle, torch.no_grad():
            for messages in batch_messages:
                prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
                gen_kwargs = dict(
                    do_sample=do_sample,
                    temperature=max(temperature, 1e-5) if do_sample else None,
                    top_p=top_p if do_sample else None,
                    num_return_sequences=n,
                    max_new_tokens=max_tokens,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
                gen = self.model.generate(**inputs, **gen_kwargs)
                prompt_len = inputs["input_ids"].shape[1]
                results.append(
                    [self.tokenizer.decode(seq[prompt_len:], skip_special_tokens=True) for seq in gen]
                )
        return results


# ─────────────────────────────────────────────────────────────────────────
# Raw FOL (pre-assembly) + assembly — the seam the runner splits on
# ─────────────────────────────────────────────────────────────────────────


@dataclass
class RecordFol:
    """One record's raw model output, BEFORE Z3 assembly. `goals` holds one
    (label, candidate_fols) entry for a Yes/No/Uncertain record (label
    '__goal__') or one per option for MCQ (label = the option text)."""

    answer_type: AnswerType
    premises_fol: list[str]
    goals: list[tuple[str, list[str]]]


def assemble_translations(
    answer_type: AnswerType,
    premises_fol: list[str],
    goals: list[tuple[str, list[str]]],
    max_skip_fraction: float = 0.25,
    ground_goals: bool = True,
    assert_type_facts: bool = True,
    type_guard_min_rules: int = 2,
) -> list[list[Translation]]:
    """Build the `list[list[Translation]]` the pipeline expects from raw FOL.

    Pure (no model): used both by single-phase `translate()` and by the runner
    AFTER predicate names have been canonicalized."""
    repair = dict(
        ground_goals=ground_goals,
        assert_type_facts=assert_type_facts,
        type_guard_min_rules=type_guard_min_rules,
    )
    if answer_type == AnswerType.YES_NO_UNCERTAIN:
        out: list[Translation] = []
        candidates = goals[0][1] if goals else []
        for i, goal_fol in enumerate(candidates):
            code, goal_expr = assemble_z3_program(premises_fol, goal_fol, max_skip_fraction, **repair)
            if code is None:
                continue
            out.append(Translation(code=code, goal_expr=goal_expr, raw_text=goal_fol, sample_index=i))
        return [out] if out else []
    if answer_type == AnswerType.MCQ:
        groups: list[list[Translation]] = []
        for _label, candidates in goals:
            per_option: list[Translation] = []
            for i, goal_fol in enumerate(candidates):
                code, goal_expr = assemble_z3_program(premises_fol, goal_fol, max_skip_fraction, **repair)
                if code is None:
                    continue
                per_option.append(
                    Translation(code=code, goal_expr=goal_expr, raw_text=goal_fol, sample_index=i)
                )
            groups.append(per_option)
        # MCQ solve needs every option to have at least one candidate program.
        return groups if (groups and all(groups)) else []
    return []


# ─────────────────────────────────────────────────────────────────────────
# Translator (drop-in for the pipeline's process_record)
# ─────────────────────────────────────────────────────────────────────────


class LlamaFolTranslator:
    """Same `translate(record) -> list[list[Translation]]` shape the pipeline
    expects, so `pipeline.process_record` runs it unchanged.

    Unlike the old T5 path, `backend` is a real chat backend (the base Llama with
    its adapter disabled), so Stage-3b CoT fallback is available."""

    def __init__(self, backend: LlamaFolBackend, cfg: LlamaFolConfig):
        self.fol_backend = backend
        # Exposed so pipeline.process_record can run the CoT fallback on the base
        # model (chat_generate(..., lora_path=None) disables the adapter).
        self.backend = backend
        self.cfg = cfg

    def _translate_one(self, sentence: str) -> list[str]:
        return self.fol_backend.translate_sentences([sentence], self.cfg.k_samples)[0]

    def _translate_many(self, sentences: list[str]) -> list[str]:
        """Best (first) FOL per sentence — used for the shared premise set."""
        groups = self.fol_backend.translate_sentences(sentences, self.cfg.k_samples)
        return [g[0] if g else "" for g in groups]

    def translate(self, record: Record) -> list[list[Translation]]:
        """Single-phase path: translate to raw FOL then assemble immediately (no
        predicate grouping). The runner instead calls translate_to_fol,
        canonicalizes names, and calls assemble_translations itself."""
        if record.answer_type == AnswerType.OPEN_ENDED:
            return []
        rfol = self.translate_to_fol(record)
        return assemble_translations(
            rfol.answer_type, rfol.premises_fol, rfol.goals, self.cfg.max_skip_fraction,
            ground_goals=self.cfg.ground_goals,
            assert_type_facts=self.cfg.assert_type_facts,
            type_guard_min_rules=self.cfg.type_guard_min_rules,
        )

    def translate_to_fol(self, record: Record) -> RecordFol:
        """Run the model to produce raw FOL only — no Z3 assembly. Phase A of the
        runner stores these before the grouping phase."""
        if record.answer_type == AnswerType.YES_NO_UNCERTAIN:
            premises_fol = self._translate_many(record.premises_nl)
            goal_candidates = self._translate_one(record.question_nl)
            rfol = RecordFol(record.answer_type, premises_fol, [("__goal__", goal_candidates)])
        elif record.answer_type == AnswerType.MCQ and record.options:
            premises_fol = self._translate_many(record.premises_nl)
            goals: list[tuple[str, list[str]]] = []
            # Feed the BARE option claim, not the "which statement can be inferred:"
            # frame — the model was trained on a single declarative statement.
            for opt in record.options:
                claim = opt.strip().rstrip(".").strip()
                goals.append((opt, self._translate_one(claim)))
            rfol = RecordFol(record.answer_type, premises_fol, goals)
        else:
            rfol = RecordFol(record.answer_type, [], [])
        self._dump_rfol(record, rfol)
        return rfol

    def _dump(self, payload: dict) -> None:
        if not self.cfg.dump_io_path:
            return
        try:
            with open(self.cfg.dump_io_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        except OSError as e:
            log.warning("could not write dump_io_path %s: %s", self.cfg.dump_io_path, e)

    def _dump_rfol(self, record: Record, rfol: RecordFol) -> None:
        if not self.cfg.dump_io_path:
            return
        premises = [
            {"nl": nl, "fol": fol, "norm": normalize_fol(fol)}
            for nl, fol in zip(record.premises_nl, rfol.premises_fol)
        ]
        if rfol.answer_type == AnswerType.MCQ:
            payload = {
                "id": record.id, "type": "mcq", "premises": premises,
                "options": [{"option": label, "fol_candidates": cands} for label, cands in rfol.goals],
            }
        else:
            cands = rfol.goals[0][1] if rfol.goals else []
            payload = {
                "id": record.id, "type": "ynu", "premises": premises,
                "question": {"nl": record.question_nl, "fol_candidates": cands},
            }
        self._dump(payload)


_GLOBAL_LLAMA_TRANSLATOR: LlamaFolTranslator | None = None


def get_llama_translator(cfg: LlamaFolConfig | None = None) -> LlamaFolTranslator:
    """Lazy singleton — one model load per process."""
    global _GLOBAL_LLAMA_TRANSLATOR
    if _GLOBAL_LLAMA_TRANSLATOR is None:
        cfg = cfg or LlamaFolConfig()
        backend = LlamaFolBackend(cfg)
        _GLOBAL_LLAMA_TRANSLATOR = LlamaFolTranslator(backend, cfg)
    return _GLOBAL_LLAMA_TRANSLATOR
