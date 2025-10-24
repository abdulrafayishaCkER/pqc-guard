# pqc_guard

A fine-tuned [Phi-2](https://huggingface.co/microsoft/phi-2) model based on **Post-Quantum Cryptography (PQC)** knowledge base areas.  
This model covers in-depth domains of **post-quantum cryptography** and various PQC scenarios.

---

## 🔗 Training Code and Artifacts
- **Kaggle Training Notebook & Output (includes zip checkpoints):** [Kaggle Link](https://www.kaggle.com/code/shahzaibali005/finetune)
- **Kaggle Model Evaluation:** [https://www.kaggle.com/code/abdulrafay07/evaluation]
- **Hugging Face Model Repository:** [https://huggingface.co/rafayishaCked/pqc_guard/]

---

## 📌 Notes
- Base model: `microsoft/phi-2`
- Fine-tuned with LoRA adapters on PQC dataset
- Intended for research & experimentation in **post-quantum cryptography reasoning**


---

## Overview

PQC-Guard was trained on a curated PQC QA corpus (~39k examples) using LoRA adapters (PEFT) on the microsoft/phi-2 base. The model aims to:

- Provide accurate, policy-aware PQC recommendations (KEMs, signatures, HPKE, TLS hybrid design, KMS, smartcards).
- Refuse out-of-scope or sensitive requests (e.g., private keys, secret material).
- Be reproducible on Kaggle/T4-style runtimes and usable for local inference via Hugging Face adapter downloads.

---

## Quick start — Installation

> Create a Python virtual environment and install required packages.

git clone https://github.com/abdulrafayishaCkER/pqc-guard
python3 -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt

Download trained LoRA adapters (recommended: Hugging Face)

Use one of these approaches:

(A) Clone HF repo (recommended)
git lfs install
git clone https://huggingface.co/abdulrafayishaCked/pqc_guard
mv pqc_guard ./adapters

(B) Programmatic download (Python)
from huggingface_hub import snapshot_download
local_dir = snapshot_download(repo_id="abdulrafayishaCked/pqc_guard", use_auth_token=True)
# move or reference local_dir as ./adapters/pqc_guard
Ok then you have can upload these the evaluation notebook to colab or kaggle and also upload the model checkpoint the latest checkpoint or overall zip file to colab or Kaggle and then use the latest checkpoint to test the model.
(Recommended): Download the lora adapters from the link given above and then turn it in zip file and directly upload it to Kaggle. You may directly use evaluation notebook as well. Just upload the zip phi-2 lora model there and use the notebook.

---






