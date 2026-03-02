import tensorflow as tf
from array_record.python import array_record_module

def read_array_record(file_path, num_records=1):
    # Create a reader for the array_record file
    reader = array_record_module.ArrayRecordReader(file_path)
    
    # Get the number of records in the file
    num_total_records = reader.num_records()
    print(f"Total records in {file_path}: {num_total_records}")
    
    # Read and print the specified number of records
    for i in range(min(num_records, num_total_records)):
        record_bytes = reader.read([i])[0]
        example = tf.train.Example()
        example.ParseFromString(record_bytes)
        
        # Extract the 'text' feature
        text = example.features.feature['text'].bytes_list.value[0].decode('utf-8')
        print(f"\n--- Record {i} ---")
        print(text)
        print("-" * 20)

if __name__ == "__main__":
    import sys
    file_path = "output_data/llama_array_record/llama-00000-of-00050.array_record"
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    
    read_array_record(file_path)
