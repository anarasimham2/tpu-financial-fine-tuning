
# 10-K Data Preprocessing for Financial Fine-Tuning

This repository provides a streamlined, professional pipeline for transforming raw **SEC 10-K filings** into a high-quality, **Chain-of-Thought (CoT)** dataset. This data is designed for fine-tuning Large Language Models (LLMs) to perform complex financial analysis.

## 📈 Overview

The project transforms unstructured 10-K text into a structured Q&A dataset where answers include a "thinking" process. This mimics the internal reasoning of a **Senior Equity Research Analyst**, focusing on:
* Key financial drivers
* Material risks and vulnerabilities
* Structural business changes

---

## 🛠 Workflow

The pipeline consists of three main automated steps:

### 1. Download Raw 10-K Documents
Fetch markdown versions of 10-K filings from the SEC EDGAR database.
```bash
python3 download_raw_10k_docs.py \
  --tickers TSLA GOOGL \
  --output_dir input_data/raw \
  --identity "Your Name <your.email@example.com>"
```

### 2. Create the QA Dataset
Uses Gemini to process raw documents, identify relevant sections (Items 1, 1A, 7, 7A, 8), and generate reasoning-heavy Q&A.
```bash
python3 create_qa_dataset_jsonl.py \
  --input_dir input_data/raw \
  --output_file input_data/dataset.jsonl \
  --workers 20
```
> **Note:** The resulting `dataset.jsonl` contains records with a `<think>...</think>` block preceding the final summary.

### 3. Convert to ArrayRecord for TPU Training
Convert the JSONL dataset into model-specific sharded `ArrayRecord` files, automatically applying chat templates (e.g., Qwen, Llama).
```bash
python3 convert_jsonl_to_arrayrecord.py \
    --input input_data/dataset.jsonl \
    --output_prefix output_data/qwen_data \
    --model_id Qwen/Qwen2.5-7B-Instruct \
    --hf_token $HF_TOKEN \
    --shards 16
```

---

## 🏗 Training & Infrastructure

### 1. Provisioning Resources
Set up the TPU VM and Google Cloud Storage (GCS) bucket.

```bash
# Set environment variables
export TPU_VM_NAME=finan-ft-v6e-vm
export PROJECT_ID=$(gcloud config get-value project)
export BUCKET_NAME=maxtext-tptft-finance

# Create TPU VM
gcloud compute tpus tpu-vm create ${TPU_VM_NAME} \
  --project=${PROJECT_ID} \
  --zone=${ZONE} \
  --accelerator-type=${ACCELERATOR_TYPE} \
  --version=${RUNTIME_VERSION}

# Create GCS Bucket
gcloud storage buckets create gs://$BUCKET_NAME --location=${REGION}
```

### 2. Environment Setup
Inside the TPU VM, configure Python 3.12 and install **MaxText**.

```bash
# Install Python 3.12
sudo apt update && sudo apt install software-properties-common -y
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update && sudo apt install python3.12 python3.12-venv -y

# Setup Virtual Environment
python3.12 -m venv ~/venv/maxtext && source ~/venv/maxtext/bin/activate

# Install MaxText
git clone https://github.com/AI-Hypercomputer/maxtext.git && cd maxtext/
```

### 3. MaxText Fine-Tuning
Convert the base model checkpoint to MaxText format and launch training.

```bash
# Convert Checkpoint
python -m MaxText.utils.ckpt_conversion.to_maxtext MaxText/configs/base.yml \
    model_name=$MODEL_NAME \
    hf_access_token=$HF_TOKEN

# Launch Training Workload
python3 -m MaxText.train MaxText/configs/base.yml \
    model_name=${MODEL_NAME} \
    base_output_directory=${BASE_OUTPUT_DIR} \
    load_parameters_path=${LOAD_PARAMETERS_PATH} \
    learning_rate=2e-5 \
    dataset_type=grain \
    grain_file_type=arrayrecord \
    grain_train_files=${TRAIN_FILES} \
    per_device_batch_size=4 \
    max_target_length=4096 \
    run_name=qwen_finetune \
    steps=700
```

---

## 🚀 Inference & Evaluation

### 1. Serve with vLLM
Once fine-tuning is complete, convert the checkpoint to Safetensors and serve via vLLM.

```bash
# Install vLLM for TPU
pip install vllm-tpu

# Serve the model
vllm serve /dev/shm/my-finetuned-model-safetensor --max-model-len=4096
```

### 2. Validation
Test the model using a standard prompt to ensure the CoT reasoning is active.

```bash
curl -X POST http://localhost:8000/v1/completions \
-H "Content-Type: application/json" \
-d '{
  "prompt": "<|im_start|>user\nAnalyze the data security risks in the PAYX 10-K segment.<|im_end|>\n",
  "temperature": 0,
  "max_tokens": 512
}'
```

---

## 📂 Project Structure

| Path | Description |
| :--- | :--- |
| `input_data/` | Raw markdown 10-K documents and `dataset.jsonl`. |
| `output_data/` | Model-specific sharded `ArrayRecord` files. |
| `utils/` | Helper scripts (e.g., `read_array_record.py`). |
| `scripts/` | Core preprocessing and conversion scripts. |

## 📋 Requirements

* **Python 3.10+** (Local) / **3.12+** (TPU)
* `google-genai`
* `edgar`
* `transformers`
* `tensorflow`
* `array-record`

> **Note:** Ensure `GEMINI_API_KEY` and `HF_TOKEN` are set in your environment.
