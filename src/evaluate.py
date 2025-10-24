#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PQC-Guard — Single-file SFT (+ optional DPO) runner for microsoft/phi-2.
Designed to run in a Kaggle/Colab-like environment. Saves LoRA SFT and optional DPO adapters,
then zips outputs under the working directory.

Copy this file to your runtime and run with:
    python run_pqc_guard.py

Notes:
 - This script enforces microsoft/phi-2 only.
 - Configure HUGGINGFACE_TOKEN in env for private HF access if required.
 - Kaggle-friendly defaults: TOKENIZERS_PARALLELISM=false, dataloader_num_workers=0.
 - Adjust DATA_FILE, OUT_SFT, OUT_DPO, and hyperparams at top as needed.
"""

import os
import sys
import subprocess
import time
import json
import math
import random
from collections import Counter
from dataclasses import dataclass
from typing import List, Dict, Optional

# ------------------ VERY EARLY ENV (must come before tokenizers/transformers imports) ------------------
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_DISABLE_TELEMETRY"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

# ------------------ USER-TUNABLE PARAMETERS ------------------
DATA_FILE = "/kaggle/input/pqc-hack/dataset.jsonl"
MODEL_NAME = "microsoft/phi-2"  # STRICT: enforced below
OUT_SFT = "/kaggle/working/pqc-phi2-lora"
OUT_DPO = "/kaggle/working/pqc-phi2-lora-dpo"
VAL_SPLIT = 0.05
MAX_LENGTH = 2048
BATCH_SIZE = 1
GRAD_ACCUM = 16
SFT_EPOCHS = 1
DPO_EPOCHS = 1
SFT_LR = 1.5e-4
DPO_LR = 5e-6
WARMUP_RATIO = 0.05
LR_SCHED = "cosine"
SEED = 42
HF_TOKEN = os.environ.get("HUGGINGFACE_TOKEN")  # set as Kaggle secret env var

SYSTEM_PROMPT = (
    "You are PQC-Guard. Stay strictly within post-quantum cryptography. "
    "If a query is outside PQC, briefly say it's out of scope."
)
USER_TAG = "<|user|>"
ASSISTANT_TAG = "<|assistant|>"
SYSTEM_TAG = "<|system|>"
END_TAG = "</s>"
REFUSAL_TEXT = (
    "I'm focused on PQC (post-quantum cryptography) topics only. "
    "For PQC algorithms, migrations, KEMs, signatures, or TLS/PKI, I can help."
)

# ------------------ helper: robust pip installs (minimal, Kaggle-friendly) ------------------
def pip_install(pkgs, retries: int = 2):
    for spec in pkgs:
        attempt = 0
        while attempt < retries:
            try:
                print(f"[pip] Installing: {spec} (attempt {attempt+1})")
                subprocess.run(
                    [sys.executable, "-m", "pip", "install", "--no-cache-dir", spec],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                )
                break
            except subprocess.CalledProcessError:
                attempt += 1
                print(f"[pip] Failed install attempt {attempt} for {spec}. Retrying...")
                time.sleep(1 + attempt)
        else:
            print(
                f"[pip] WARNING: Could not install {spec} after {retries} attempts. Proceeding anyway."
            )

# minimal required packages (avoid optional heavy extras)
to_install = [
    "transformers>=4.34.0",
    "datasets",
    "accelerate",
    "peft",
    "trl",
    "einops",
    "bitsandbytes>=0.45.3",
]

# ------------------ main entry ------------------
def main():
    # install runtime deps (best-effort)
    pip_install(to_install)

    # ------------------ imports (after installs) ------------------
    import torch
    from datasets import Dataset
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        BitsAndBytesConfig,
        Trainer,
        TrainingArguments,
        set_seed,
        TrainerCallback,
        TrainerState,
        TrainerControl,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import DPOTrainer

    # ------------------ environment & determinism ------------------
    set_seed(SEED)
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
    except Exception:
        pass
    USE_CUDA = torch.cuda.is_available()
    try:
        device_name = torch.cuda.get_device_name(0) if USE_CUDA else "CPU"
    except Exception:
        device_name = "CPU"
    print(f"[ENV] Device: {device_name} | Torch {torch.__version__} | CUDA available={USE_CUDA}")

    # ------------------ load data (resilient) ------------------
    rows: List[Dict] = []
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"Data file not found: {DATA_FILE} (set DATA_FILE correctly)")

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    modes = Counter((r.get("response") or {}).get("mode") for r in rows)
    missing_ans = sum(
        1
        for r in rows
        if not isinstance((r.get("response") or {}).get("answer"), str)
        or not (r.get("response") or {}).get("answer").strip()
    )
    print(f"[DATA] Loaded {len(rows)} rows | Mode counts: {dict(modes)} | Empty 'answer' rows: {missing_ans}")

    # ------------------ prompt helpers ------------------
    def build_prompt(instr: str, kb_refs: Optional[List[str]] = None) -> str:
        ctx = f"Context kb_refs: {', '.join(kb_refs)}\n" if kb_refs else ""
        return (
            f"{SYSTEM_TAG}\n{SYSTEM_PROMPT}\n"
            f"{USER_TAG}\nInstruction: {instr if instr else '[no instruction provided]'}\n{ctx}"
            f"{ASSISTANT_TAG}\n"
        )

    def row_to_text(x: Dict) -> Optional[Dict]:
        instr = (x.get("instruction") or "").strip()
        ctx = x.get("context") or {}
        kb_refs = None
        if isinstance(ctx, dict) and isinstance(ctx.get("kb_refs"), list) and ctx["kb_refs"]:
            kb_refs = [str(k) for k in ctx["kb_refs"]]
        resp = x.get("response") or {}
        ans = (resp.get("answer") or "").strip()
        if not ans:
            ans = REFUSAL_TEXT
        prompt = build_prompt(instr, kb_refs)
        full = prompt + ans + "\n" + END_TAG
        return {"prompt": prompt, "answer": ans, "text": full}

    mapped = []
    for r in rows:
        rec = row_to_text(r)
        if rec:
            mapped.append(rec)

    print("[DATA] Mapped rows:", len(mapped))
    full_ds = Dataset.from_list(mapped)
    split = (
        full_ds.train_test_split(test_size=VAL_SPLIT, seed=SEED)
        if len(full_ds) > 1
        else {"train": full_ds, "test": Dataset.from_list([])}
    )
    train_ds, val_ds = split["train"], split["test"]

    # ------------------ tokenizer ------------------
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, use_auth_token=HF_TOKEN)
    added = tokenizer.add_special_tokens(
        {"additional_special_tokens": [USER_TAG, ASSISTANT_TAG, SYSTEM_TAG]}
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    print(f"[TOKENIZER] added {added} special tokens; pad={tokenizer.pad_token!r}")

    # ------------------ tokenization with label masking ------------------
    def tokenize_with_mask(ex):
        prompt = ex["prompt"]
        full = ex["text"]
        ft = tokenizer(full, truncation=True, max_length=MAX_LENGTH)
        pt = tokenizer(prompt, truncation=True, max_length=MAX_LENGTH)
        input_ids = ft["input_ids"]
        attention_mask = ft["attention_mask"]
        labels = input_ids.copy()
        prompt_len = len(pt["input_ids"]) if isinstance(pt["input_ids"], list) else len(pt["input_ids"][0])
        for i in range(min(prompt_len, len(labels))):
            labels[i] = -100
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    # Use num_proc=1 to avoid tokenizers + map forking warnings on Kaggle
    train_tok = train_ds.map(tokenize_with_mask, remove_columns=train_ds.column_names, num_proc=1)
    val_tok = (
        val_ds.map(tokenize_with_mask, remove_columns=val_ds.column_names, num_proc=1)
        if len(val_ds) > 0
        else None
    )
    print(
        "[TOKENIZE] Tokenized datasets:",
        len(train_tok),
        " / val:",
        len(val_tok) if val_tok else 0,
    )

    # ------------------ BitsAndBytes config ------------------
    bnb_compute_dtype = (
        torch.bfloat16
        if (USE_CUDA and torch.cuda.get_device_capability(0)[0] >= 8)
        else torch.float16
    )
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=bnb_compute_dtype,
    )

    # ------------------ Strict model-loading (PHI-2 only) ------------------
    def check_enforce_phi2(name: str) -> str:
        if name != "microsoft/phi-2":
            print(f"[ENFORCE] Overriding MODEL_NAME -> 'microsoft/phi-2' (user or env had: {name})")
        return "microsoft/phi-2"

    nonlocal_model_name = MODEL_NAME  # keep original var's name for inner scope
    MODEL_NAME_LOCAL = check_enforce_phi2(nonlocal_model_name)

    def try_load_model_phi2(name: str, use_4bit: bool = True):
        kwargs = {"trust_remote_code": True}
        if HF_TOKEN:
            kwargs["use_auth_token"] = HF_TOKEN
        if torch.cuda.is_available():
            kwargs["device_map"] = "auto"
        try:
            if use_4bit:
                print(f"[MODEL] Attempting 4-bit load for {name} ...")
                model = AutoModelForCausalLM.from_pretrained(name, quantization_config=bnb_config, **kwargs)
            else:
                print(f"[MODEL] Attempting fp16 load for {name} ...")
                # fp16 fallback on same model only
                torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
                model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch_dtype, **kwargs)
            # sanity: ensure model reports correct name/path in config
            reported = getattr(model.config, "_name_or_path", None) or getattr(model, "name_or_path", None)
            print(f"[MODEL] Loaded model -> reported name: {reported}")
            if reported and "phi-2" not in str(reported).lower() and "phi2" not in str(reported).lower():
                # if model doesn't look like phi2, raise (we refuse other models)
                raise RuntimeError(
                    f"Loaded model does not look like phi-2 (reported: {reported}). Aborting by design."
                )
            return model
        except Exception as e:
            print(f"[MODEL] Failed to load {name} (use_4bit={use_4bit}): {type(e).__name__}: {e}")
            return None

    # Try 4-bit first, then fp16 — but only for PHI-2. If both fail, raise and stop.
    base_model = try_load_model_phi2(MODEL_NAME_LOCAL, use_4bit=True)
    if base_model is None:
        print("[MODEL] 4-bit load failed, trying fp16 fallback for phi-2 (same model only).")
        base_model = try_load_model_phi2(MODEL_NAME_LOCAL, use_4bit=False)

    if base_model is None:
        raise RuntimeError(
            "Unable to load microsoft/phi-2 on this runtime. By design we DO NOT auto-fallback to other models. "
            "Common causes: insufficient GPU memory, missing HF token, or network/access issues. "
            "Options: (1) ensure HUGGINGFACE_TOKEN is set and has access; (2) pick a runtime with more GPU memory; "
            "or (3) manually load a smaller phi-2 variant if available. Aborting."
        )

    # Resize embeddings to account for added tokens
    base_model.resize_token_embeddings(len(tokenizer))
    base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=True)
    print("[MODEL] prepare_model_for_kbit_training complete.")

    # ------------------ LoRA config & apply ------------------
    linear_names = {n.split(".")[-1] for n, m in base_model.named_modules() if isinstance(m, torch.nn.Linear)}
    CANDS = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "Wqkv",
        "out_proj",
        "fc1",
        "fc2",
    ]
    target_modules = sorted([t for t in CANDS if t in linear_names]) or sorted(list(linear_names))[:8]
    print("[LORA] LoRA targets:", target_modules)

    peft_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    policy = get_peft_model(base_model, peft_cfg)
    policy.config.use_cache = False
    policy.config.pad_token_id = tokenizer.pad_token_id

    # ------------------ collator (no worker forks) ------------------
    @dataclass
    class LMDataCollator:
        tokenizer: any

        def __call__(self, feats: List[Dict]):
            import torch as _torch

            pad_id = self.tokenizer.pad_token_id
            ids = [_torch.tensor(f["input_ids"], dtype=_torch.long) for f in feats]
            amask = [_torch.tensor(f["attention_mask"], dtype=_torch.long) for f in feats]
            labs = [_torch.tensor(f["labels"], dtype=_torch.long) for f in feats]
            input_ids = _torch.nn.utils.rnn.pad_sequence(ids, batch_first=True, padding_value=pad_id)
            attention_mask = _torch.nn.utils.rnn.pad_sequence(amask, batch_first=True, padding_value=0)
            labels = _torch.nn.utils.rnn.pad_sequence(labs, batch_first=True, padding_value=-100)
            return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    collator = LMDataCollator(tokenizer)

    # ------------------ Trainer progress callback (logs memory & ensures model is PHI-2) ------------------
    class ProgressCallback(TrainerCallback):
        def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
            # Called frequently; keep output minimal but informative
            step = state.global_step
            if torch.cuda.is_available():
                used = torch.cuda.memory_allocated() / (1024 ** 2)
                reserved = torch.cuda.memory_reserved() / (1024 ** 2)
                print(f"[PROG] step={step} | cuda_used={used:.1f}MB reserved={reserved:.1f}MB")
                # try nvidia-smi quick query (best-effort)
                try:
                    out = subprocess.run(
                        ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.DEVNULL,
                        text=True,
                        timeout=2,
                    )
                    if out.returncode == 0:
                        print("[PROG] nvidia-smi:", out.stdout.strip())
                except Exception:
                    pass
            else:
                print(f"[PROG] step={step} (CPU run)")

        def on_save(self, args, state: TrainerState, control: TrainerControl, **kwargs):
            # verify saved model is phi-2 (best-effort)
            model = kwargs.get("model") or policy
            reported = getattr(model.config, "_name_or_path", None) or getattr(model, "name_or_path", None)
            print(f"[PROG] save triggered at step {state.global_step}. reported model: {reported}")
            if reported and "phi-2" not in str(reported).lower() and "phi2" not in str(reported).lower():
                print("[PROG][WARNING] Saved model does not look like phi-2. THIS SHOULD NOT HAPPEN (enforced).")

    # ------------------ SFT training (Trainer) ------------------
    os.makedirs(OUT_SFT, exist_ok=True)
    sft_args = TrainingArguments(
        output_dir=OUT_SFT,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=SFT_EPOCHS,
        learning_rate=SFT_LR,
        lr_scheduler_type=LR_SCHED,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=0.01,
        max_grad_norm=0.3,
        bf16=(USE_CUDA and torch.cuda.get_device_capability(0)[0] >= 8),
        fp16=(USE_CUDA and torch.cuda.get_device_capability(0)[0] < 8),
        logging_steps=50,
        eval_strategy="steps" if val_tok else "no",
        eval_steps=500,
        save_strategy="steps",
        save_steps=800,
        save_total_limit=2,
        report_to="none",
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        group_by_length=True,
        dataloader_num_workers=0,  # avoid worker forks on Kaggle
    )

    trainer = Trainer(
        model=policy,
        args=sft_args,
        train_dataset=train_tok,
        eval_dataset=val_tok if val_tok else None,
        data_collator=collator,
        tokenizer=tokenizer,
        callbacks=[ProgressCallback()],
    )

    print("[SFT] Starting SFT training (phi-2 only). If this fails due to OOM, consider using a runtime with more GPU memory.")
    trainer.train()
    trainer.model.save_pretrained(OUT_SFT)
    tokenizer.save_pretrained(OUT_SFT)
    print(f"[SFT] Saved LoRA SFT -> {OUT_SFT}")

    # evaluate basic eval loss if available (best-effort)
    try:
        m = trainer.evaluate()
        if (lv := m.get("eval_loss")) is not None:
            ppl = math.exp(lv) if lv < 20 else float("inf")
            print(f"[SFT] eval_loss={lv:.4f}, ppl={ppl:.2f}")
    except Exception as e:
        print("[SFT] evaluation skipped or failed:", e)

    # ------------------ DPO dataset + training (auto-skip) ------------------
    pairs_flat = []
    for r in rows:
        p = r.get("preference_pair")
        if isinstance(p, dict) and {"prompt", "chosen", "rejected"} <= set(p.keys()):
            rec = {"prompt": str(p["prompt"]), "chosen": str(p["chosen"]), "rejected": str(p["rejected"])}
            if isinstance(p.get("kb_refs"), list):
                rec["kb_refs"] = p["kb_refs"]
            pairs_flat.append(rec)

    if len(pairs_flat) == 0:
        print("[DPO] No preference pairs found → DPO skipped.")
    else:
        def map_pairs(p):
            kb = p.get("kb_refs") if isinstance(p.get("kb_refs"), list) else None
            ctx = f"Context kb_refs: {', '.join(kb)}\n" if kb else ""
            prompt = (
                f"{SYSTEM_TAG}\n{SYSTEM_PROMPT}\n"
                f"{USER_TAG}\nInstruction: {p.get('prompt','')}\n{ctx}"
                f"{ASSISTANT_TAG}\n"
            )
            return {"prompt": prompt, "chosen": str(p.get("chosen", "")), "rejected": str(p.get("rejected", ""))}

        tmp_pairs = Dataset.from_list(pairs_flat)
        dpo_ds = tmp_pairs.map(
            map_pairs,
            remove_columns=[c for c in tmp_pairs.column_names if c not in {"prompt", "chosen", "rejected"}],
            num_proc=1,
        )
        print("[DPO] DPO rows:", len(dpo_ds))

        # load *reference* model for DPO — must be phi-2 too. Try fp16 only for reference.
        ref_model = try_load_model_phi2(MODEL_NAME_LOCAL, use_4bit=False)
        if ref_model is None:
            print("[DPO] Couldn't load phi-2 reference model for DPO; skipping DPO stage by design.")
        else:
            os.makedirs(OUT_DPO, exist_ok=True)
            dpo_args = TrainingArguments(
                output_dir=OUT_DPO,
                per_device_train_batch_size=BATCH_SIZE,
                per_device_eval_batch_size=BATCH_SIZE,
                gradient_accumulation_steps=GRAD_ACCUM,
                num_train_epochs=DPO_EPOCHS,
                learning_rate=DPO_LR,
                lr_scheduler_type=LR_SCHED,
                warmup_ratio=WARMUP_RATIO,
                bf16=(USE_CUDA and torch.cuda.get_device_capability(0)[0] >= 8),
                fp16=(USE_CUDA and torch.cuda.get_device_capability(0)[0] < 8),
                logging_steps=50,
                save_strategy="steps",
                save_steps=800,
                save_total_limit=2,
                report_to="none",
                optim="paged_adamw_8bit",
                group_by_length=True,
                dataloader_num_workers=0,
            )

            dpo_trainer = DPOTrainer(
                model=policy,
                ref_model=ref_model,
                args=dpo_args,
                beta=0.1,
                train_dataset=dpo_ds,
                tokenizer=tokenizer,
            )
            print("[DPO] Starting DPO training (phi-2 reference).")
            dpo_trainer.train()
            dpo_trainer.model.save_pretrained(OUT_DPO)
            tokenizer.save_pretrained(OUT_DPO)
            print("[DPO] Saved LoRA DPO ->", OUT_DPO)

    # ------------------ Zip outputs ------------------
    import zipfile

    def zip_dir(src_dir: str, zip_path: str):
        src_dir = os.path.abspath(src_dir)
        with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(src_dir):
                for f in files:
                    full = os.path.join(root, f)
                    rel = os.path.relpath(full, src_dir)
                    zf.write(full, arcname=rel)

    sft_zip = "/kaggle/working/pqc-phi2-lora.zip"
    zip_dir(OUT_SFT, sft_zip)
    print("[ZIP] Zipped:", sft_zip)

    if os.path.isdir(OUT_DPO) and any(True for _ in os.scandir(OUT_DPO)):
        dpo_zip = "/kaggle/working/pqc-phi2-lora-dpo.zip"
        zip_dir(OUT_DPO, dpo_zip)
        print("[ZIP] Zipped:", dpo_zip)
    else:
        print("[ZIP] No DPO output to zip (skipped).")

    print(
        "[DONE] All done. Check /kaggle/working for outputs. NOTE: This run strictly enforces microsoft/phi-2 only (no automatic fallbacks to other models)."
    )

if __name__ == "__main__":
    main()
