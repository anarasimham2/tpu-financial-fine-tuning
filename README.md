# 10-K Data Preprocessing for Financial Fine-Tuning

This repository provides a streamlined, professional pipeline for transforming raw SEC 10-K filings into a high-quality, chain-of-thought (CoT) dataset suitable for fine-tuning Large Language Models (LLMs) on financial analysis tasks.

## Overview

The goal of this project is to take the complex, unstructured text of 10-K filings and generate a structured Q&A dataset where each answer is preceded by a "thinking" process. This mimics the internal reasoning of a Senior Equity Research Analyst, focusing on key financial drivers, risks, and material changes.

## Workflow

The pipeline consists of three main steps:

### 1. Download Raw 10-K Documents
Use the `download_raw_10k_docs.py` script to fetch markdown versions of 10-K filings from the SEC EDGAR database.

```bash
python3 download_raw_10k_docs.py --tickers TSLA GOOGL --output_dir input_data/raw --identity "Your Name <your.email@example.com>"
```

### 2. Create the QA Dataset
The `create_qa_dataset_jsonl.py` script uses Gemini to process the raw documents. It identifies relevant sections (e.g., Item 1, 1A, 7, 7A, 8), chunks the text, and generates both a professional summary and a corresponding "internal thinking" block.

```bash
python3 create_qa_dataset_jsonl.py --input_dir input_data/raw --output_file input_data/dataset.jsonl --workers 20
```
*The resulting `dataset.jsonl` contains records with a `<think>...</think>` block followed by the final summary.*

### 3. Convert to ArrayRecord for TPU Training
Finally, convert the JSONL dataset into model-specific sharded `ArrayRecord` files using `convert_jsonl_to_arrayrecord.py`. This script automatically applies the appropriate chat template for your target model (e.g., Qwen, Llama).

```bash
python3 convert_jsonl_to_arrayrecord.py \
    --input input_data/dataset.jsonl \
    --output_prefix output_data/qwen_data \
    --model_id Qwen/Qwen2.5-7B-Instruct \
    --hf_token $HF_TOKEN \
    --shards 16
```

## Project Structure

- `input_data/`: Contains raw markdown docs and the final `dataset.jsonl`.
- `output_data/`: Contains the model-specific, sharded `ArrayRecord` files ready for training.
- `utils/`: Includes helper scripts, such as `read_array_record.py` for inspecting processed data.

## Training & Inference Infrastructure

### 1. Setup the TPU VM
Set up the TPU VM, GCS Bucket, and the required packages.

```bash
export TPU_VM_NAME=finan-ft-v6e-vm

gcloud compute tpus tpu-vm create ${TPU_VM_NAME} \
  --project=${PROJECT_ID} \
  --zone=${ZONE} \
  --accelerator-type=${ACCELERATOR_TYPE} \
  --version=${RUNTIME_VERSION}

gcloud compute tpus tpu-vm ssh ${TPU_VM_NAME} \
       --zone=us-central1-a \
       --project=<project_id>
```

### 2. Set up GCS Bucket
```bash
export BUCKET_NAME=maxtext-tptft-finance
gcloud storage buckets create gs://$BUCKET_NAME --location=${REGION}
```

### 3. Install the required packages
Set noninteractive mode for Debian package installer:

```bash
sudo sed -i.bak -e "s/^#*\s*\$nrconf{restart}\s*=\s*['\"][il]['\"];/\$nrconf{restart} = 'a';/" /etc/needrestart/needrestart.conf

export NEEDRESTART_SUSPEND=1
export NEEDRESTART_MODE=1
export DEBIAN_FRONTEND=noninteractive
```

Install python3.12 as MaxText requires python>=3.12:

```bash
sudo apt update && sudo apt install software-properties-common -y && sudo add-apt-repository ppa:deadsnakes/ppa && sudo apt update && sudo apt install python3.12 python3.12-venv -y
```


## MaxText Configuration & Training

### 1. Environment setup on Cloud TPUs
Create a venv environment and install the necessary packages:

```bash
python3.12 -m venv ~/venv/maxtext && source ~/venv/maxtext/bin/activate
git clone https://github.com/AI-Hypercomputer/maxtext.git && cd maxtext/
```

### 2. Configuring the MaxText parameters for this dataset
Convert the pre-trained model checkpoint (e.g., Qwen3-8b) into the MaxText-compatible format:

```bash
python -m MaxText.utils.ckpt_conversion.to_maxtext MaxText/configs/base.yml \
    model_name=$MODEL_NAME \
    hf_access_token=$HF_TOKEN
```

### 3. Execution commands for the fine-tuning run
Set up the env variables in the TPU VM for the fine-tuning job with output_data from data generated earlier:

```bash
export HF_TOKEN=<HF_TOKEN>
export PROJECT_ID=$(gcloud config get-value project)
export REGION=us-east5
export ZONE=us-east5-a
export ACCELERATOR_TYPE=v6e-8
export RUNTIME_VERSION=v2-alpha-tpuv6e

export MODEL_NAME=qwen3-8b 
export TOKENIZER_CONFIG="tokenizer_path=Qwen/Qwen3-8B"
export BASE_OUTPUT_DIR=gs://$BUCKET_NAME/output-qwen
export DATASET_TYPE=grain 
export TRAIN_FILES=/output_data/qwen_data/qwen_array_record/qwen-*.array_record 
export LOAD_PARAMETERS_PATH=gs://$BUCKET_NAME/pretrained-ckpt-orbax-qwen/0/items
export STEPS=700
export CHECKPOINT_PERIOD=350
```

Launch the fine-tuning workload:

```bash
python3 -m MaxText.train MaxText/configs/base.yml \
    model_name=${MODEL_NAME} \
    base_output_directory=${BASE_OUTPUT_DIR} \
    load_parameters_path=${LOAD_PARAMETERS_PATH} \
    learning_rate=2e-5 \
    attention=flash \
    num_vocab_tiling=8 \
    dataset_type=${DATASET_TYPE} \
    grain_file_type=arrayrecord \
    hf_access_token=${HF_TOKEN} \
    grain_train_files=${TRAIN_FILES} \
    ${TOKENIZER_CONFIG} \
    per_device_batch_size=4 \
    remat_policy=full \
    max_target_length=4096 \
    ici_fsdp_parallelism=1 \
    ici_tensor_parallelism=-1 \
    run_name=qwen_finetune \
    steps=${STEPS} \
    checkpoint_period=${CHECKPOINT_PERIOD} \
    async_checkpointing=true \
    scan_layers=true
```

When the fine-tuning is finished, convert the fine-tuned Orbax checkpoint into a Safetensors checkpoint:

```bash
python3 -m MaxText.utils.ckpt_conversion.to_huggingface MaxText/configs/base.yml \
    model_name=$MODEL_NAME \
    hf_access_token=$HF_TOKEN \
    load_parameters_path=gs://$BUCKET_NAME/output/my_finetune_runner/checkpoints/29/items \
    base_output_directory=/dev/shm/my-finetuned-model-safetensor \
    scan_layers=true
```


### Inference & Evaluation

1. Setup for inference
*Install TPU vLLM*
```bash
pip install vllm-tpu
```

*Start the vLLM with the fine-tuned weights from the previous step*
```bash
vllm serve /dev/shm/my-finetuned-model-safetensor --max-model-len=4096 > ~/vllm.out 2>&1 &
tail -f ~/vllm.out
```
Example output:
```text
.....
INFO:     Started server process [1]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

2. Evaluation
*Once the vLLM starts successfully, validate with a simple prompt to validate the response*
```bash
USER_PROMPT="What is the tallest building in Seoul?"

curl -X POST http://localhost:8000/v1/completions \
-H "Content-Type: application/json" \
-d @- <<EOF
{
"prompt": "<|im_start|>user\n${USER_PROMPT}<|im_end|>\n",
"temperature": 0,
"max_tokens": 128,
"top_p": 1.0
}
EOF
```

*Specific prompt to try to find difference between Base and Fine-Tuned model*

Analyze this segment from `PAYX_10k.md`:

> Data Security and Privacy Leaks: We collect, use, and retain increasingly large amounts of personal information about our clients, employees of our clients, our employees, and other third parties, including: bank account, credit card, and social security numbers, tax return information, health care information, retirement account information, payroll information, system and network passwords, and other sensitive personal and business information. At the same time, the continued occurrence of high-profile cyber and ransomware attacks and data breaches provides evidence of an external environment increasingly hostile to information security. We may be particularly targeted for cyberattack because of the amount and type of personal and business information that we collect, use, and retain, as well as during and after periods in which we acquire other companies. Vulnerabilities, threats, and more sophisticated and targeted computer crimes pose a risk to the security of our systems and networks, and the confidentiality, availability, and integrity of our data. Furthermore, if any of our solutions contain a software vulnerability, the vulnerability may be exploited to obtain access to our data or our clients’ data.

## Requirements

- Python 3.10+
- `google-genai`
- `edgar`
- `transformers`
- `tensorflow`
- `array-record`

Ensure your `GEMINI_API_KEY` is set in a `.env` file or exported in your environment.
