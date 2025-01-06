#!/bin/bash

# Directory containing JSON files
DATA_DIR="/mnt/petrelfs/hujucheng/train/data/70Wbase"

# Base directory for saving results
SAVE_BASE_DIR="/mnt/petrelfs/hujucheng/train/data/70Wbase_IFD_scores"

# Loop through each JSON file in the directory
for data_file in "$DATA_DIR"/*.json; do
    # Extract the base name of the file (without directory and extension)
    base_name=$(basename "$data_file" .json)
    
    # Construct the save path
    save_path="${SAVE_BASE_DIR}/${base_name}_IFD_cherry.pt"
    
    # Create a temporary script for sbatch
    sbatch_script=$(mktemp)
    
    cat <<EOT > "$sbatch_script"
#!/bin/bash
#SBATCH --partition=belt_road_102B
#SBATCH --job-name=sampling_by_IFD_scoring
#SBATCH --quotatype=reserved
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5

python /mnt/petrelfs/hujucheng/SFT_tools/Cherry_LLM/cherry_seletion/data_analysis.py \
--data_path "$data_file" \
--save_path "$save_path" \
--model_name_or_path /mnt/petrelfs/hujucheng/models/internlm2_5_7b_chat_cpt_hu200k_base70w/20241202114239/hf-581 \
--max_length 512 \
--prompt openai \
--mod cherry
EOT

    # Submit the job
    sbatch "$sbatch_script"
    
    # Remove the temporary script
    rm "$sbatch_script"
    
    # Sleep for 2 seconds
    sleep 2
done