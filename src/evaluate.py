#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pqc_guard_eval.py
Single-file evaluation runner that:
  1) Loads microsoft/phi-2 base model and runs a small prompt set (deterministic / sampling).
  2) Loads LoRA adapters from a local directory and runs the same prompt set per adapter.
  3) Saves JSON results to /kaggle/working/pqc_phi2_eval_results.json (adjustable).

Designed to run in Kaggle / Colab / local environments with CUDA. Installs minimal packages if needed.
Set HUGGINGFACE_TOKEN in env if private model access is required.

Usage:
  python pqc_guard_eval.py \
    --adapters_dir /kaggle/input/finetuned/pqc-phi2-lora \
    --output /kaggle/working/pqc_phi2_eval_results.json
"""

from __future__ import annotations
import argparse
import os
import sys
import subprocess
import json
import traceback
import time
from typing import List, Dict, Optional

# -----------------------
# Minimal runtime installs (best-effort)
# -----------------------
REQ_PKGS = ["transformers>=4.35.0", "accelerate", "sentencepiece", "safetensors", "peft"]

def pip_install(pkgs: List[str]):
    for p in pkgs:
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "--no-cache-dir", p],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            # best-effort: continue if install fails
            print(f"[pip] Warning: install failed for {p}, continuing anyway.", file=sys.stderr)

# attempt imports; if missing, install then re-import
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
    from peft import PeftModel
except Exception:
    print("[ENV] Missing packages; attempting to pip install requirements (this may take a while).")
    pip_install(REQ_PKGS)
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList
    from peft import PeftModel

# -----------------------
# Config / defaults
# -----------------------
DEFAULT_MODEL_ID = "microsoft/phi-2"
DEFAULT_ADAPTERS_DIR = "/kaggle/input/finetuned/pqc-phi2-lora"
DEFAULT_OUTPUT = "/kaggle/working/pqc_phi2_eval_results.json"
HF_TOKEN = os.environ.get("HUGGINGFACE_TOKEN")

SYSTEM_TAG = "<|system|>"
USER_TAG = "<|user|>"
ASSISTANT_TAG = "<|assistant|>"
END_TAG = "</s>"

SYSTEM_PROMPT = (
    "You are PQC-Guard. Stay strictly within post-quantum cryptography. "
    "If a query is outside PQC, briefly say it's out of scope."
)

PROMPT_DICT = {
    "q1": "Explain backpropagation in Transformers.",
    "q6": "We encrypt backups with ChaCha20-Poly1305 + wrap DEKs using RSA-2048. Is this quantum-resistant-secure?",
    "q7": "Our TLS uses X448 and TLS_ECDHE_ECDSA_ with _AES_256_GCM_SHA384. Is this quantum-resistant?",
    "q13": "What is Kyber?",
    "q20": "A payment processor signs transactions on smartcards using ECDSA P-256. What is the PQC transition plan?",
    "q35": "Bank migration: how do we shift from ECC to post-quantum in three steps while keeping legacy clients working?",
    # add more keys if desired...
}

# -----------------------
# Utilities: prompt builder + cleaner
# -----------------------
def build_prompt(instr: str, kb_refs: Optional[List[str]] = None) -> str:
    ctx = f"Context kb_refs: {', '.join(kb_refs)}\n" if kb_refs else ""
    return (
        f"{SYSTEM_TAG}\n{SYSTEM_PROMPT}\n"
        f"{USER_TAG}\nInstruction: {instr}\n{ctx}"
        f"{ASSISTANT_TAG}\n"
    )

def clean_text(raw: Optional[str], prompt: str) -> str:
    if raw is None:
        return "<empty>"
    t = raw.replace("\r", "")
    if prompt and prompt in t:
        t = t.replace(prompt, " ")
    for tok in (SYSTEM_TAG, USER_TAG, ASSISTANT_TAG, "<|endoftext|>", "<s>", "</s>"):
        t = t.replace(tok, " ")
    import re
    t = re.sub(r"<\|[^>]*\|>", " ", t)
    t = re.sub(r"\n\s+\n", "\n\n", t)
    t = re.sub(r"[ \t]{2,}", " ", t)
    lines = [ln.rstrip() for ln in t.splitlines()]
    cleaned = []
    prev = None
    for l in lines:
        if l == prev:
            continue
        if l.strip() == "" and prev == "":
            continue
        cleaned.append(l)
        prev = l
    out = "\n".join(cleaned).strip()
    out = re.sub(r"(\s*</s>\s*)+", "", out)
    out = out.strip()
    return out or "<empty-response>"

# -----------------------
# Stopping criteria: stop if assistant tag repeats or new role appears
# -----------------------
class StopOnRoleRepeat(StoppingCriteria):
    def __init__(self, tokenizer, prompt_len_tokens):
        super().__init__()
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len_tokens

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        seq = input_ids[0].tolist()
        try:
            text = self.tokenizer.decode(seq, skip_special_tokens=False)
        except Exception:
            return False
        if text.count(ASSISTANT_TAG) >= 2:
            return True
        first_assistant_pos = text.find(ASSISTANT_TAG)
        if first_assistant_pos != -1:
            tail = text[first_assistant_pos + len(ASSISTANT_TAG):]
            if USER_TAG in tail or SYSTEM_TAG in tail:
                return True
        return False

# -----------------------
# Adapter discovery: pick largest numeric or newest
# -----------------------
def pick_largest_checkpoint(base_dir: str) -> List[str]:
    if not os.path.isdir(base_dir):
        return []
    subdirs = [d for d in sorted(os.listdir(base_dir)) if os.path.isdir(os.path.join(base_dir, d))]
    if not subdirs:
        return [base_dir]
    import re
    numeric_dirs = [d for d in subdirs if re.fullmatch(r'\d+', d)]
    if numeric_dirs:
        chosen = max(numeric_dirs, key=lambda x: int(x))
        return [os.path.join(base_dir, chosen)]
    dir_int_pairs = []
    for d in subdirs:
        nums = re.findall(r'\d+', d)
        if nums:
            dir_int_pairs.append((d, max(int(n) for n in nums)))
    if dir_int_pairs:
        chosen = max(dir_int_pairs, key=lambda x: x[1])[0]
        return [os.path.join(base_dir, chosen)]
    subdirs_with_mtime = [(d, os.path.getmtime(os.path.join(base_dir, d))) for d in subdirs]
    chosen = max(subdirs_with_mtime, key=lambda x: x[1])[0]
    return [os.path.join(base_dir, chosen)]

# -----------------------
# Main evaluation flows
# -----------------------
def eval_base_model(model_id: str, prompts: Dict[str, str], max_new_tokens: int = 150):
    """Load base phi-2 and run prompts, returning dict of answers."""
    print(f"[BASE] Loading base model: {model_id}")
    kwargs = {}
    if HF_TOKEN:
        kwargs["use_auth_token"] = HF_TOKEN
    try:
        if torch.cuda.is_available():
            print("[BASE] CUDA available -> using torch.float16 and device_map='auto'")
            model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto", trust_remote_code=False, **kwargs)
        else:
            print("[BASE] CUDA not available -> loading on CPU")
            model = AutoModelForCausalLM.from_pretrained(model_id, device_map={"": "cpu"}, trust_remote_code=False, **kwargs)
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False, **({"use_auth_token": HF_TOKEN} if HF_TOKEN else {}))
    except Exception as e:
        print("[BASE] Primary load failed; attempting CPU fallback (slower). Exception:", e)
        traceback.print_exc()
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map={"": "cpu"})
    model.eval()
    results = {}
    for k, q in prompts.items():
        prompt = build_prompt(q)
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {kk: vv.to(next(model.parameters()).device) for kk, vv in inputs.items()}
        gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=True, temperature=0.2, top_p=0.9,
                          repetition_penalty=1.05, eos_token_id=tokenizer.eos_token_id)
        try:
            with torch.no_grad():
                out = model.generate(**inputs, **gen_kwargs)
            text = tokenizer.decode(out[0], skip_special_tokens=True)
            if text.startswith(prompt):
                text = text[len(prompt):].strip()
            ans = clean_text(text, prompt)
        except Exception as e:
            ans = f"<generation-error: {e}>"
        print(f"[BASE][{k}] -> {ans[:200].replace(chr(10),' ')}")
        results[k] = ans
    return {"model_id": model_id, "device": "cuda" if torch.cuda.is_available() else "cpu", "answers": results}

def eval_adapters(model_id: str, adapters_dir: str, prompts: Dict[str, str], max_new_tokens: int = 512):
    """For each discovered adapter checkpoint, load base model then apply adapter and generate prompts."""
    out = {}
    checkpoints = pick_largest_checkpoint(adapters_dir)
    if not checkpoints:
        print(f"[ADAPTERS] No adapters found in {adapters_dir}")
        return out
    print(f"[ADAPTERS] Checkpoints discovered: {checkpoints}")
    tokenizer = None
    # prefer loading tokenizer from adapter if possible
    for cand in checkpoints + [adapters_dir]:
        try:
            tokenizer = AutoTokenizer.from_pretrained(cand, use_fast=False)
            print(f"[TK] loaded tokenizer from {cand}")
            break
        except Exception:
            continue
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
        print("[TK] fallback tokenizer loaded from base model")

    # ensure special tokens
    tokenizer.add_special_tokens({"additional_special_tokens":[USER_TAG, ASSISTANT_TAG, SYSTEM_TAG]})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    for cp in checkpoints:
        print("\n" + "="*60)
        print("[ADAPTER] Processing:", cp)
        try:
            base = AutoModelForCausalLM.from_pretrained(model_id, device_map={"": "cpu"}, trust_remote_code=True)
            if getattr(base.config, "vocab_size", None) != len(tokenizer):
                base.resize_token_embeddings(len(tokenizer))
        except Exception as e:
            print("[ERR] base load failed:", e)
            traceback.print_exc()
            continue

        try:
            model = PeftModel.from_pretrained(base, cp, device_map={"": "cpu"})
            model.config.use_cache = False
            model.eval()
        except Exception as e:
            print("[ERR] peft apply failed:", e)
            traceback.print_exc()
            # try to load cp as a full model fallback
            try:
                model = AutoModelForCausalLM.from_pretrained(cp, device_map={"": "cpu"}, trust_remote_code=True)
                model.eval()
            except Exception as e2:
                print("[ERR] fallback failed:", e2)
                traceback.print_exc()
                continue

        device_name = "cpu"
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                model.to("cuda")
                try:
                    model.half()
                except Exception:
                    pass
                device_name = "cuda"
                print("[MOVE] model on GPU")
            except Exception as e:
                print("[MOVE] keep on CPU due to:", e)
                model.to("cpu")

        cp_res = {}
        for k, q in prompts.items():
            prompt = build_prompt(q)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            inputs = {kk: vv.to(next(model.parameters()).device) for kk, vv in inputs.items()}
            # stopping criteria to avoid extra role blocks
            prompt_len = inputs["input_ids"].shape[1]
            stopcrit = StopOnRoleRepeat(tokenizer=tokenizer, prompt_len_tokens=prompt_len)
            stopping_criteria = StoppingCriteriaList([stopcrit])
            gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=False, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id, stopping_criteria=stopping_criteria)
            try:
                with torch.no_grad():
                    out = model.generate(**inputs, **gen_kwargs)
                raw = tokenizer.decode(out[0], skip_special_tokens=False)
                ans = clean_text(raw, prompt)
            except Exception as e:
                ans = f"<generation-error: {e}>"
            print(f"[{k}] -> {ans[:200].replace(chr(10),' ')}")
            cp_res[k] = ans
        out[cp] = {"device": device_name, "answers": cp_res}
    return out

# -----------------------
# CLI
# -----------------------
def parse_args():
    p = argparse.ArgumentParser(description="PQC-Guard evaluation runner (base model + adapters).")
    p.add_argument("--model_id", type=str, default=DEFAULT_MODEL_ID, help="Base model id (microsoft/phi-2)")
    p.add_argument("--adapters_dir", type=str, default=DEFAULT_ADAPTERS_DIR, help="Directory containing LoRA adapter subfolders")
    p.add_argument("--output", type=str, default=DEFAULT_OUTPUT, help="Output JSON path")
    p.add_argument("--max_tokens_base", type=int, default=150, help="Max new tokens for base model generation")
    p.add_argument("--max_tokens_adapter", type=int, default=512, help="Max new tokens for adapter generation")
    return p.parse_args()

def main():
    args = parse_args()
    # run base model eval
    base_res = eval_base_model(args.model_id, PROMPT_DICT, max_new_tokens=args.max_tokens_base)

    # run adapter eval
    adapters_res = eval_adapters(args.model_id, args.adapters_dir, PROMPT_DICT, max_new_tokens=args.max_tokens_adapter)

    out = {
        "meta": {"model_id": args.model_id, "adapters_dir": args.adapters_dir, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
        "base": base_res,
        "adapters": adapters_res,
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"[SAVE] Results saved to {args.output}")

if __name__ == "__main__":
    main()
