import re
import json
import time
import os
import glob
import argparse
import threading
import random
import hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from google import genai
from google.genai import types
from tqdm import tqdm

# ==========================================
# CONFIGURATION
# ==========================================
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

MODEL_PRIORITY = [
    "gemini-3-flash-preview",
    "gemini-2.5-pro",
    "gemini-3-pro-preview",
    "gemini-2.5-flash"
]

RELEVANT_ITEMS = ["ITEM 1.", "ITEM 1A.", "ITEM 7.", "ITEM 7A.", "ITEM 8."]

# State Management
stop_event = threading.Event()
write_lock = threading.Lock()
exhausted_models = set()
exhausted_lock = threading.Lock()

def generate_chunk_hash(text):
    normalized = re.sub(r'\s+', ' ', text.strip())
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()

def append_to_jsonl(entry, output_file):
    with write_lock:
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

def mark_model_exhausted(model_name):
    with exhausted_lock:
        if model_name not in exhausted_models:
            print(f"\n[QUOTA HIT] {model_name} exhausted.")
            exhausted_models.add(model_name)

def generate_analyst_response(chunk_data, client):
    raw_text, filename = chunk_data
    if stop_event.is_set(): return None

    summary_sys = "You are a Senior Equity Research Analyst. Rewrite the segment into a concise, professional summary."
    thinking_sys = "You are a Senior Equity Research Analyst. Provide the internal chain-of-thought for the summary. Wrap in <think> tags."

    for model_name in MODEL_PRIORITY:
        with exhausted_lock:
            if model_name in exhausted_models: continue
        try:
            s_resp = client.models.generate_content(
                model=model_name,
                config=types.GenerateContentConfig(system_instruction=summary_sys, temperature=0.7),
                contents=f"Source: {filename}\n\n{raw_text}"
            )
            summary = s_resp.text
            t_resp = client.models.generate_content(
                model=model_name,
                config=types.GenerateContentConfig(system_instruction=thinking_sys, temperature=0.7),
                contents=f"10-K: {raw_text}\n\nSummary: {summary}"
            )
            thinking = t_resp.text.strip()
            if "<think>" not in thinking: thinking = f"<think>\n{thinking}\n</think>"
            elif "</think>" not in thinking: thinking = f"{thinking}\n</think>"
            return f"{thinking}\n\n{summary}"
        except Exception as e:
            if "quota" in str(e).lower(): mark_model_exhausted(model_name)
            continue
    return None

def parse_markdown_sections(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    parts = re.split(r'(^##\s+(?:Item|ITEM)\s+[0-9A-Z\.]+.*?$)', content, flags=re.MULTILINE)
    sections, current_header = {}, None
    for part in parts:
        part = part.strip()
        if not part: continue
        if part.upper().startswith("## ITEM"):
            match = re.search(r'(?:Item|ITEM)\s+[0-9A-Z\.]+', part, re.IGNORECASE)
            if match: current_header = match.group(0).upper() 
        elif current_header:
            sections[current_header] = part
            current_header = None
    return sections

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default="input_data/raw")
    parser.add_argument("--output_file", default="input_data/dataset.jsonl")
    parser.add_argument("--limit_files", type=int, default=None)
    parser.add_argument("--limit_chunks", type=int, default=None)
    parser.add_argument("--workers", type=int, default=20)
    args = parser.parse_args()

    client = genai.Client(api_key=API_KEY)
    files = glob.glob(os.path.join(args.input_dir, "*.md"))
    if args.limit_files: files = files[:args.limit_files]

    all_tasks = []
    for file_path in files:
        file_name = os.path.basename(file_path)
        sections = parse_markdown_sections(file_path)
        for section, content in sections.items():
            if section in RELEVANT_ITEMS:
                sub_chunks = content.split("\n\n")
                curr = ""
                for para in sub_chunks:
                    if len(curr) + len(para) < 1500: curr += para + "\n\n"
                    else:
                        if len(curr) > 400: all_tasks.append((curr, file_name))
                        curr = para + "\n\n"

    if args.limit_chunks:
        all_tasks = all_tasks[:args.limit_chunks]

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(generate_analyst_response, t, client): t for t in all_tasks}
        for f in tqdm(as_completed(futures), total=len(all_tasks)):
            raw, fname = futures[f]
            res = f.result()
            if res:
                append_to_jsonl({
                    "messages": [
                        {"role": "system", "content": "You are a financial analyst helper."},
                        {"role": "user", "content": f"Analyze this segment from {fname}:\n\n{raw.strip()}"},
                        {"role": "assistant", "content": res}
                    ]
                }, args.output_file)

if __name__ == "__main__":
    main()
