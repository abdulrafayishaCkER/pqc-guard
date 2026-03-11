# 🚀 PQC-Guard

[![Python](https://img.shields.io/badge/python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Other-lightgrey)](LICENSE)
[![Hugging Face](https://img.shields.io/badge/🤗%20Model-rafayishaCked%2Fpqc__guard-yellow)](https://huggingface.co/rafayishaCked/pqc_guard)

A fine-tuned model based on **[Phi-2](https://huggingface.co/microsoft/phi-2)**, specialized in **Post-Quantum Cryptography (PQC)**.  
This model provides enhanced reasoning across PQC domains and is suitable for research, testing, and experimentation.

---

## 📚 Description

**PQC-Guard** is trained on a curated **Post-Quantum Cryptography QA dataset (~39k samples)** using **LoRA (PEFT)** on the `microsoft/phi-2` base model.

### ✅ Features

- Accurate and reliable PQC recommendations (KEMs, Signatures, HPKE, Hybrid PQ-TLS, KMS, Smartcards, Hardware Integration)
- Policy-aware responses (no private keys, exploits, or sensitive material)
- Lightweight, reproducible, and easy to run on Kaggle, Google Colab, or locally

---

## 🔗 Resources

| Resource | Link |
|----------|-------|
| 🧠 Training Notebook + Checkpoints | [kaggle.com/code/shahzaibali005/finetune](https://www.kaggle.com/code/shahzaibali005/finetune) |
| 🧪 Model Evaluation Notebook | [kaggle.com/code/abdulrafay07/evaluation](https://www.kaggle.com/code/abdulrafay07/evaluation) |
| 🤗 Hugging Face Model Repository | [huggingface.co/rafayishaCked/pqc_guard](https://huggingface.co/rafayishaCked/pqc_guard/) |

---

## 🧩 Model Details

- **Base Model:** `microsoft/phi-2`
- **Fine-Tuning Method:** LoRA (PEFT)
- **Dataset Size:** ~39,000 PQC QA pairs
- **Intended Use:** Research & experimentation in **Post-Quantum Cryptography**

---

## 🗂️ Project Structure

```
pqc-guard/
├── data/
│   └── dataset_generator.py   # PQC Q&A dataset generator
├── src/
│   ├── train_lora.py           # LoRA fine-tuning script
│   └── evaluate.py             # Evaluation runner
├── checkpoints/
│   └── README.md               # Checkpoint hosting info
├── notebooks/                  # Kaggle/Colab notebooks
├── results/                    # Evaluation outputs
├── requirements.txt
├── CITATION.cff
└── LICENSE
```

---

## ⚙️ Installation & Setup

Clone the repository, create a virtual environment, and install dependencies:

```bash
git clone https://github.com/abdulrafayishaCkER/pqc-guard
cd pqc-guard

python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

### Download LoRA Adapters

Choose one method:

**A) Clone from Hugging Face (Recommended)**

```bash
git lfs install
git clone https://huggingface.co/rafayishaCked/pqc_guard
mv pqc_guard ./adapters
```

**B) Programmatic Download using Python**

```python
from huggingface_hub import snapshot_download

local_dir = snapshot_download(
    repo_id="rafayishaCked/pqc_guard",
    use_auth_token=True
)
# Move or reference `local_dir` as ./adapters/pqc_guard
```

---

## 🧪 Evaluation

You can upload the evaluation notebook and LoRA adapters to Kaggle or Google Colab.

### Recommended Workflow

1. Download the LoRA adapters from Hugging Face
2. Zip the adapter folder and upload it to Kaggle/Colab
3. Load the latest checkpoint inside the evaluation notebook
4. Run inference + model testing

You may also use the evaluation notebook directly by uploading the zipped Phi-2 LoRA model.

### Recommendations

- Use the LoRA adapter version for best compatibility
- Prefer running on GPU (T4 Tesla)
- Ideal for PQC research, model comparison, and hybrid PQ-TLS design testing

---

## 📄 Citation

If you use this project, please cite it using the information in [`CITATION.cff`](CITATION.cff):

```
PQC-Guard: Fine-Tuning Phi-2 for Post-Quantum Cryptography Readiness
Muhammad Masoom Alam, Abdul Rafay — 2025
https://github.com/abdulrafayishaCkER/pqc-guard
```

