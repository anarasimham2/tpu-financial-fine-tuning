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

*Co-author TODO: Document the infrastructure setup here.*

### MaxText Configuration & Training
*Co-author TODO: Provide the specific MaxText steps, including:*
1.  Environment setup on Cloud TPUs.
2.  Configuring the `MaxText` parameters for this dataset.
3.  Execution commands for the fine-tuning run.

### Inference & Evaluation
*Co-author TODO: Detail the steps for running inference and evaluating the fine-tuned model's performance on financial tasks.*

## Requirements

- Python 3.10+
- `google-genai`
- `edgar`
- `transformers`
- `tensorflow`
- `array-record`

Ensure your `GEMINI_API_KEY` is set in a `.env` file or exported in your environment.
