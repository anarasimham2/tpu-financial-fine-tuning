import argparse
import json
import glob
import os
import tensorflow as tf
from array_record.python import array_record_module
from transformers import AutoTokenizer

def create_tf_example(text_content):
    """Wraps the raw text string into a tf.train.Example Key-Value pair."""
    return tf.train.Example(features=tf.train.Features(feature={
        'text': tf.train.Feature(bytes_list=tf.train.BytesList(value=[text_content.encode('utf-8')]))
    }))

def convert_and_flatten(input_pattern, output_prefix, num_shards, model_id, hf_token):
    print(f"📥 Loading Tokenizer for: {model_id}...")
    try:
        # Load the specific tokenizer (Qwen, Llama, Mistral, etc.)
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
        
        # Validation: Check if this model actually has a chat template
        if not tokenizer.chat_template:
            print(f"⚠️  WARNING: The model '{model_id}' does not have a default chat_template defined.")
            print("   If this is a Base model, this script might fail or produce raw JSON strings.")
        else:
            print(f"✅ Chat Template found! Using template for: {model_id}")

    except Exception as e:
        print(f"❌ Error loading tokenizer: {e}")
        return

    input_files = glob.glob(input_pattern)
    if not input_files:
        print(f"❌ No input files found matching: {input_pattern}")
        return

    # Create writers for each shard
    writers = []
    for i in range(num_shards):
        path = f"{output_prefix}-{i:05d}-of-{num_shards:05d}.array_record"
        writers.append(array_record_module.ArrayRecordWriter(path, 'group_size:1'))

    total_count = 0
    skipped_count = 0
    
    try:
        for input_file in input_files:
            print(f"Processing file: {input_file}...")
            with open(input_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    if not line.strip(): continue
                    
                    try:
                        data = json.loads(line)
                        # Expecting a key 'entry' which is a list of dicts
                        messages = data.get('messages', None)
                        
                        if not messages or not isinstance(messages, list):
                            skipped_count += 1
                            continue
                        
                        # --- THE MAGIC STEP ---
                        # apply_chat_template automatically handles:
                        # - Llama 3: <|start_header_id|>...
                        # - Qwen: <|im_start|>...
                        # - Mistral: [INST]...
                        flattened_text = tokenizer.apply_chat_template(
                            messages, 
                            tokenize=False,  # Keep as string for ArrayRecord
                            add_generation_prompt=False 
                        )
                        # ----------------------

                        example = create_tf_example(flattened_text)
                        
                        # Round-robin write to shards
                        writers[total_count % num_shards].write(example.SerializeToString())
                        total_count += 1

                        if total_count % 5000 == 0:
                            print(f"   Converted {total_count} records...", end='\r')

                    except Exception as e:
                        print(f"Error on line {line_num} in {input_file}: {e}")
                        skipped_count += 1

    finally:
        for w in writers: w.close()
            
    print(f"\n\n🎉 Finished!")
    print(f"   - Total Records: {total_count}")
    print(f"   - Skipped/Invalid: {skipped_count}")
    print(f"   - Output: {output_prefix}-*.array_record")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert JSONL chat data to ArrayRecord with Model-Specific Templates")
    
    parser.add_argument("--input", required=True, help="Input glob pattern (e.g. input_data/*.jsonl)")
    parser.add_argument("--output_prefix", required=True, help="Output file prefix (e.g. output_data/qwen_data)")
    parser.add_argument("--model_id", required=True, help="HuggingFace Model ID (e.g. Qwen/Qwen2.5-7B-Instruct)")
    parser.add_argument("--hf_token", required=True, help="Hugging Face Access Token")
    parser.add_argument("--shards", type=int, default=16, help="Number of output shards (default: 16)")

    args = parser.parse_args()

    convert_and_flatten(
        args.input, 
        args.output_prefix, 
        args.shards, 
        args.model_id, 
        args.hf_token
    )
