🚀 PQC-Guard

A fine-tuned model based on Phi-2
, specialized in Post-Quantum Cryptography (PQC).
This model provides enhanced PQC reasoning across various cryptographic domains.

📚 Description

PQC-Guard was trained on a curated PQC QA dataset (~39k samples) using LoRA adapters (PEFT) on the microsoft/phi-2 base model.
The model focuses on:

✅ Accurate PQC reasoning and recommendations (KEMs, signatures, HPKE, PQ-TLS, hybrid schemes, KMS, hardware tokens)
✅ Policy-aware responses (avoids private keys, exploits, confidential material)
⚙️ Easy to run on Kaggle / Colab / local inference with Hugging Face adapter support

🔗 Resources
Resource	Link
🧠 Training Notebook + Checkpoints	https://www.kaggle.com/code/shahzaibali005/finetune

🧪 Model Evaluation Notebook	https://www.kaggle.com/code/abdulrafay07/evaluation

🤗 Hugging Face Model Repository	https://huggingface.co/rafayishaCked/pqc_guard/
🧩 Model Details

Base Model: microsoft/phi-2

Fine-tuning Method: LoRA (PEFT)

Dataset Size: ~39,000 PQC QA pairs

Use Case: Research & experiments in post-quantum cryptography

⚙️ Installation & Setup

Create a Python virtual environment and install dependencies:

git clone https://github.com/abdulrafayishaCkER/pqc-guard
cd pqc-guard

python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt

📥 Download LoRA Adapters

Choose one of the following approaches:

A) Clone from Hugging Face (Recommended)
git lfs install
git clone https://huggingface.co/abdulrafayishaCked/pqc_guard
mv pqc_guard ./adapters

B) Programmatic Download with Python
from huggingface_hub import snapshot_download

local_dir = snapshot_download(
    repo_id="abdulrafayishaCked/pqc_guard",
    use_auth_token=True
)

# Move or reference `local_dir` as ./adapters/pqc_guard

🧪 Evaluation

You may upload the evaluation notebook and model checkpoint to Colab or Kaggle.
Recommended workflow:

Download the LoRA adapters (from HF link above)

Zip them and upload to Kaggle/Colab

Load the latest checkpoint inside the evaluation notebook

Run inference tests

⭐ Recommendations

✅ Use the LORA adapter version for best compatibility
💡 "Evaluation Notebook" can be used directly—just upload the zipped Phi-2 LoRA model
