import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import jsonlines

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(
    "/mnt/hwfile/opendatalab/peiqizhi/checkpoints/internlm2_5_7b_chat_reduced_ds_v1_70w/20241118163419/hf-581", 
    trust_remote_code=True
)
model = AutoModelForCausalLM.from_pretrained(
    "/mnt/hwfile/opendatalab/peiqizhi/checkpoints/internlm2_5_7b_chat_reduced_ds_v1_70w/20241118163419/hf-581", 
    torch_dtype=torch.float16, 
    trust_remote_code=True
).cuda()
model = model.eval()

# Read inputs from JSON file
# input_file = "/mnt/petrelfs/hujucheng/SFT_tools/SFT_toolkit/data/reading_comperhensive/SFT_data_for_LSAT_RC_from_trian_1797.json"  # Input JSON file path
# output_file = "/mnt/petrelfs/hujucheng/SFT_tools/SFT_toolkit/data/embeddings_of_train/SFT_data_for_LSAT_RC_from_trian_1797.json"  

input_file = "/mnt/petrelfs/hujucheng/SFT_tools/SFT_toolkit/data/benchmark_set/hu_MCQ_logiqa-hu.json" 
output_file = "/mnt/petrelfs/hujucheng/SFT_tools/SFT_toolkit/data/embeddings_of_test/hu_MCQ_logiqa.jsonl"  
#Note the output would be a jsonl and need to be parse back to json

def extract_from_test():
    with open(input_file, "r") as f:
        data = json.load(f)  # Assumes data is a list of dictionaries
    
    cnt = 0
    total = len(data)

    with jsonlines.open(output_file, mode='w') as writer:
        for entry in data:
            cnt += 1
            # text = entry["messages"][0]["content"]
            text = entry["passage"]+"\n"+entry["question"]+"\n"+ "".join(entry["options"])
            # Tokenize and forward pass to get embeddings
            inputs = tokenizer(text, return_tensors="pt").to("cuda")
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                embeddings = outputs.hidden_states[-1][:, 0, :].squeeze().cpu().tolist()  # CLS token embedding
            
            # Add embeddings to the JSON entry
            entry["embedding"] = embeddings
            
            # Write each entry to the JSONL file
            writer.write(entry)
            
            if cnt % 100 == 0:
                print(f"{cnt} processed, {total-cnt} remaining")


extract_from_test()