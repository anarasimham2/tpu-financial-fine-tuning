
# 10-K Data Preprocessing for Financial Fine-Tuning

This repository provides a streamlined, professional pipeline for transforming raw **SEC 10-K filings** into a high-quality, **Chain-of-Thought (CoT)** dataset. This data is designed for fine-tuning Large Language Models (LLMs) to perform complex financial analysis.

This repository supports the associated [fine-tuning blog post](https://medium.com/@ashish.narasimham/financial-analysis-across-a-10-k-document-corpus-language-fine-tuning-with-tpu-and-maxtext-6d57a3d0eda2).

## 📈 Overview

The project transforms unstructured 10-K text into a structured Q&A dataset where answers include a "thinking" process. This mimics the internal reasoning of a **Senior Equity Research Analyst**, focusing on:
* Key financial drivers
* Material risks and vulnerabilities
* Structural business changes

---

## 🛠 Synthetic Data Preparation

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

## 🏗 Fine Tuning with Jax and Maxtext

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

### 4. Convert fine-tuned Orbax checkpoint into Safetensors checkpoint
To facilitate the subsequent inference stage in TPU vLLM, in this we convert the Orbax checkpoint into Safetensors checkpoint

 ```bash
python3 -m MaxText.utils.ckpt_conversion.to_huggingface MaxText/configs/base.yml \
    model_name=$MODEL_NAME \
    hf_access_token=$HF_TOKEN \
    load_parameters_path=gs://$BUCKET_NAME/output/my_finetune_runner/checkpoints/29/items \
    base_output_directory=/dev/shm/my-finetuned-model-safetensor \ scan_layers=true
```

---

## 🚀 TPU vLLM Inference & Evaluation

### 1. Serve with TPU vLLM
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

### 3. Sample response

One specific prompt we tried:

```text
Analyze this segment from PAYX_10k.md:

Data Security and Privacy Leaks: We collect, use, and retain increasingly large amounts of personal information about our clients, employees of our clients, our employees, and other third parties, including: bank account, credit card, and social security numbers, tax return information, health care information, retirement account information, payroll information, system and network passwords, and other sensitive personal and business information. At the same time, the continued occurrence of high-profile cyber and ransomware attacks and data breaches provides evidence of an external environment increasingly hostile to information security. We may be particularly targeted for cyberattack because of the amount and type of personal and business information that we collect, use, and retain, as well as during and after periods in which we acquire other companies. Vulnerabilities, threats, and more sophisticated and targeted computer crimes pose a risk to the security of our systems and networks, and the confidentiality, availability, and integrity of our data. Furthermore, if any of our solutions contain a software vulnerability, the vulnerability may be exploited to obtain access to our data or our clients’ data.
```

Here's the base model's response:

```text
<think>
Okay, let's take a look at this segment from the PAYX_10k.md. The user wants an analysis of this part. First, I need to understand what the text is about. It's talking about data security and privacy leaks. The company collects a lot of personal and sensitive information from various parties. They mention things like bank account details, credit cards, social security numbers, tax returns, healthcare info, retirement accounts, payroll data, passwords, and other sensitive info.

Then they talk about the external environment being hostile due to high-profile cyberattacks and ransomware. They note that the company might be a target because of the volume and type of data they handle, especially during acquisitions. They also mention vulnerabilities, threats, and sophisticated cybercrimes as risks. Plus, software vulnerabilities in their solutions could lead to data breaches.

So, the user probably wants an analysis of the company's data security risks and how they're addressing them. Maybe they're looking for a summary of the key points, the implications, or how this affects the company's operations and compliance. They might also be interested in the potential consequences of these risks, like financial loss, reputational damage, legal issues, or regulatory penalties.

I should break down the main points: the types of data collected, the external threats, the reasons for being a target, the risks from vulnerabilities, and the potential consequences. Also, maybe the company's acknowledgment of these risks and any measures they might be taking, though the text doesn't mention specific mitigation strategies. The user might be looking for an evaluation of the company's risk management in this area.

I need to make sure the analysis covers all these aspects. Maybe start by summarizing the key elements, then discuss the implications, and perhaps touch on the regulatory landscape (like GDPR, CCPA) even though it's not mentioned here. Also, note that the company is aware of the risks and is likely taking steps to mitigate them, even if not detailed in the text. The user might be a stakeholder, investor, or someone assessing the company's security posture.
</think>

The segment from PAYX_10k.md highlights the company's acknowledgment of significant **data security and privacy risks** in its operations. Here's a structured analysis of the key points and implications:

—

### **1. Scope of Data Collected**
- **Sensitive Information Types**: The company handles a wide range of **highly sensitive data**, including:
  - Financial data (bank accounts, credit cards, tax returns).
  - Personal identifiers (social security numbers, health care information).
  - Business-related data (retirement accounts, payroll, system passwords).
  - Third-party data (clients, employees, and other stakeholders).
- **Implication**: This extensive data collection increases the **potential impact of a breach**, as attackers could exploit this information for identity theft, financial fraud, or other malicious activities.

—

### **2. External Threat Landscape**
- **Cybersecurity Risks**: The text emphasizes the **increasing frequency and sophistication of cyberattacks**, ransomware, and data breaches. These threats are described as part of a \"hostile external environment.\"
- **Targeted Vulnerabilities**: The company is **particularly vulnerable** due to:
  - The **volume and sensitivity** of data it processes.
  - **Acquisition activities**, which may introduce new vulnerabilities or expand attack surfaces.
  - **Software vulnerabilities** in its solutions, which could be exploited to access internal or client data.

---

### **3. Risks to Systems and Data**
- **Threats to Security Pillars**: The segment underscores risks to the **confidentiality, integrity, and availability** of data:
  - **Confidentiality**: Unauthorized access to sensitive information.
  - **Integrity**: Data tampering or corruption.
  - **Availability**: System outages or ransomware disrupting operations.
- **Operational and Reputational Impact**: A breach could lead to **financial losses**, **legal liabilities**, **regulatory penalties**, and **loss of customer trust**.

---

### **4. Regulatory and Compliance Considerations**
- **Implicit Risks**: While not explicitly stated, the mention of sensitive data (e.g., health care, social security numbers) suggests potential **regulatory scrutiny** under laws like:
  - **GDPR** (for EU data subjects).
  - **CCPA** (for California residents).
  - **HIPAA** (for health care data).
- **Compliance Challenges**: The company must ensure **data protection measures** align with evolving regulations, which could require significant investment in security infrastructure and incident response protocols.

---

### **5. Mitigation and Risk Management**
- **Acknowledgment of Risks**: The company explicitly recognizes these risks, indicating a **proactive stance** in identifying vulnerabilities.
- **Potential Mitigation Strategies** (implied but not detailed):
  - **Enhanced cybersecurity measures** (e.g., encryption, multi-factor authentication).
  - **Regular vulnerability assessments** and patch management.
 "
```

Here's the fine-tuned model's response:

```
<think>
To analyze this 10-K segment, I first identified the core business function described: the collection and retention of highly sensitive, multi-jurisdictional data (SSNs, tax, health, and payroll). I then evaluated the external threat landscape, noting the explicit mention of ransomware and the company's specific vulnerability as a high-value target due to its data volume. I connected the mention of \"acquisitions\" to the operational risk of integrating legacy systems, which often introduces technical debt and security gaps. Finally, I synthesized these factors to determine the material impact on the company's risk profile, focusing on the potential for regulatory penalties, reputational damage, and the financial burden of remediation following a breach.
</think>

**Investment Memo: Data Security & Cybersecurity Risk Profile**

**Key Financial Drivers**
*   **Data Sensitivity:** Paychex’s business model relies on the collection and retention of high-value, sensitive data, including SSNs, tax returns, health care information, and retirement account details. The integrity of this data is a primary driver of client trust and long-term retention.
*   **M&A Integration:** The company’s growth strategy through acquisitions introduces significant financial risk, as integrating legacy systems and data sets often reveals latent vulnerabilities that require immediate remediation and capital expenditure.

**Risks**
*   **Targeted Cyberattacks:** Due to the volume and type of sensitive information held, Paychex is a high-value target for ransomware and sophisticated cybercriminals.
*   **Systemic Vulnerabilities:** The company faces risks from both internal software vulnerabilities and external threats to its network infrastructure, which could compromise the confidentiality, availability, and integrity of client data.
*   **Operational Disruption:** A successful breach could lead to significant reputational damage, regulatory fines, and the potential loss of clients, particularly during periods of system instability.

**Material Changes**
*   **Evolving Threat Landscape:** Management notes an increasingly hostile external environment characterized by high-profile, sophisticated ransomware attacks. 
*   **Heightened Acquisition Risk:** There is a specific emphasis on the heightened risk profile associated with post-acquisition periods, where the integration of disparate IT environments may temporarily weaken the company’s security posture."
```

Even from the shorter thinking block, you can see the brevity in the fine-tuned response while interpreting rather than summarizing the input data.

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

