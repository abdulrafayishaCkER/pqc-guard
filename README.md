# 🚀 PQC-Guard

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
|----------|------|
| 🧠 Training Notebook + Checkpoints | [Kaggle – Finetune](https://www.kaggle.com/code/shahzaibali005/finetune) |
| 🧪 Model Evaluation Notebook | [Kaggle – Evaluation](https://www.kaggle.com/code/abdulrafay07/evaluation) |
| 🤗 Hugging Face Model Repository | [rafayishaCked/pqc_guard](https://huggingface.co/rafayishaCked/pqc_guard/) |

---

## 🧩 Model Details

- **Base Model:** `microsoft/phi-2`
- **Fine-Tuning Method:** LoRA (PEFT)
- **Dataset Size:** ~39,000 PQC QA pairs
- **Intended Use:** Research & experimentation in **Post-Quantum Cryptography**

---

## 📁 Repository Structure

```
pqc-guard/
├── CITATION.cff              # Citation metadata
├── LICENSE                    # MIT License
├── README.md                  # This file
├── requirements.txt           # Python dependencies
├── checkpoints/               # Placeholder (checkpoints hosted on Hugging Face)
├── data/
│   └── dataset_generator.py   # PQC QA dataset generator script
├── notebooks/
│   ├── training.ipynb         # Training notebook
│   └── evaluation.ipynb       # Evaluation notebook
├── report/
│   └── PQC_Guard_Report_v1.docx  # Project report
├── results/
│   ├── avg_scores.png         # Average evaluation scores chart
│   ├── per_question_comparison.png  # Per-question comparison chart
│   └── pqc_evaluation.csv     # Evaluation results (CSV)
└── src/
    ├── train_lora.py          # SFT (+ optional DPO) training script
    └── evaluate.py            # Evaluation runner script
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/abdulrafayishaCkER/pqc-guard.git
cd pqc-guard
```

### 2. Create a Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate        # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 📥 Download LoRA Adapters

Choose one of the following methods:

### Option A — Clone from Hugging Face (Recommended)

```bash
git lfs install
git clone https://huggingface.co/rafayishaCked/pqc_guard
mv pqc_guard ./adapters
```

### Option B — Programmatic Download using Python

```python
from huggingface_hub import snapshot_download

local_dir = snapshot_download(
    repo_id="rafayishaCked/pqc_guard",
    use_auth_token=True
)

# Move or reference `local_dir` as ./adapters/pqc_guard
```

---

## 🏋️ Training

The training script (`src/train_lora.py`) is designed for **Kaggle / Google Colab** environments. It performs LoRA-based SFT fine-tuning on `microsoft/phi-2` with optional DPO training.

```bash
python src/train_lora.py
```

> **Note:** By default, the script expects the dataset at `/kaggle/input/pqc-hack/dataset.jsonl`. Update the `DATA_FILE` variable at the top of the script if running locally.

### Key Training Parameters (configurable in `src/train_lora.py`)

| Parameter | Default |
|-----------|---------|
| `DATA_FILE` | `/kaggle/input/pqc-hack/dataset.jsonl` |
| `MODEL_NAME` | `microsoft/phi-2` |
| `MAX_LENGTH` | `2048` |
| `BATCH_SIZE` | `1` |
| `GRAD_ACCUM` | `16` |
| `SFT_EPOCHS` | `1` |
| `SFT_LR` | `1.5e-4` |
| `SEED` | `42` |

---

## 📊 Evaluation

The evaluation script (`src/evaluate.py`) loads the base `microsoft/phi-2` model and LoRA adapters, then runs a set of PQC-related prompts to compare outputs.

```bash
python src/evaluate.py \
    --adapters_dir ./adapters/pqc_guard \
    --output ./results/pqc_phi2_eval_results.json
```

### CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--model_id` | `microsoft/phi-2` | Base model identifier |
| `--adapters_dir` | `/kaggle/input/finetuned/pqc-phi2-lora` | Path to LoRA adapter directory |
| `--output` | `/kaggle/working/pqc_phi2_eval_results.json` | Output JSON file path |
| `--max_tokens_base` | `150` | Max new tokens for base model generation |
| `--max_tokens_adapter` | `512` | Max new tokens for adapter generation |

---

## 📓 Notebooks

You can also use the provided Jupyter notebooks for training and evaluation:

1. **`notebooks/training.ipynb`** — Step-by-step training workflow
2. **`notebooks/evaluation.ipynb`** — Evaluation and comparison of base vs. fine-tuned model

### Recommended Notebook Workflow

1. Download the LoRA adapters from [Hugging Face](https://huggingface.co/rafayishaCked/pqc_guard/)
2. Zip the adapter folder and upload it to Kaggle or Google Colab
3. Load the latest checkpoint inside the evaluation notebook
4. Run inference and model testing

---

## 🗂️ Dataset Generation

To generate or extend the PQC QA dataset:

```bash
python data/dataset_generator.py
```

> By default, this generates **30,000** QA pairs and writes them to `qa_pqc_dataset.jsonl`. See the script's CLI options (`TOTAL`, `OUT`, `VARIANTS_PER_PROMPT`, `SEED`, `STYLE`, `LANG`, `SHARD_SIZE`) for customization.

---

## 💡 Recommendations

- Use the **LoRA adapter** version for best compatibility
- Prefer running on **GPU** (NVIDIA T4 Tesla or better recommended)
- Ideal for PQC research, model comparison, and hybrid PQ-TLS design testing
- Set the `HUGGINGFACE_TOKEN` environment variable if accessing private models

---

## 📄 Citation

```bibtex
@misc{alam2025pqcguard,
  author = {Muhammad Masoom Alam and Abdul Rafay},
  title = {PQC-Guard: Fine-Tuning Phi-2 for Post-Quantum Cryptography Readiness},
  year = {2025},
  url = {https://github.com/abdulrafayishaCkER/pqc-guard}
}
```

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).