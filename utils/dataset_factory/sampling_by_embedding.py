import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM

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

input_file = "/mnt/petrelfs/hujucheng/SFT_tools/SFT_toolkit/data/logic_reasoning/SFT_data_for_LSAT_LR_from_trian_3960.json"  # Input JSON file path
output_file = "/mnt/petrelfs/hujucheng/SFT_tools/SFT_toolkit/data/embeddings_of_train/SFT_data_for_LSAT_LR_from_trian_3960.json"  

def extract_from_test():

    with open(input_file, "r") as f:
        data = json.load(f)  # Assumes data is a list of dictionaries

    # Process each entry in the JSON
    for entry in data:

        # text = entry["passage"]+"\n"+entry["question"]+"\n"+ "".join(entry["options"])
        text = entry["messages"][0]["content"]
        
        # Tokenize and forward pass to get embeddings
        inputs = tokenizer(text, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            embeddings = outputs.hidden_states[-1][:, 0, :].squeeze().cpu().tolist()  # CLS token embedding
        
        # Add embeddings to the JSON entry
        entry["embedding"] = embeddings

    # Save updated data to output JSON file
    with open(output_file, "w") as f:
        json.dump(data, f, indent=4,ensure_ascii=False)

    print(f"Embeddings added and saved to {output_file}")
extract_from_test()