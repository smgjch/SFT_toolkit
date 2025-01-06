#!/bin/bash

# Directory containing JSON files
SCORE_DATA_DIR="/mnt/petrelfs/hujucheng/train/data/70Wbase_IFD_scores"

RAW_DATA_DIR="/mnt/petrelfs/hujucheng/train/data/70Wbase"

# Base directory for saving results
SAVE_BASE_DIR="/mnt/petrelfs/hujucheng/train/data/10Wbase/21WbaseIFDSelected"

# Loop through each JSON file in the directory
for data_file in "$RAW_DATA_DIR"/*.json; do
    # Extract the base name of the file (without directory and extension)
    base_name=$(basename "$data_file" .json)
    
    # Construct the save path
    pt_data_file="${SCORE_DATA_DIR}/${base_name}_IFD_cherry.pt"
    unselected_data_path="${RAW_DATA_DIR}/${base_name}.json"
    selected_data_path="${SAVE_BASE_DIR}/${base_name}_IFD_selected.json"
    # Create a temporary script for sbatch
    sbatch_script=$(mktemp)
    
    cat <<EOT > "$sbatch_script"
#!/bin/bash
#SBATCH --partition=belt_road
#SBATCH --job-name=Selection_by_IFD_score
#SBATCH --quotatype=reserved
#SBATCH --gres=gpu:0
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10

python /mnt/petrelfs/hujucheng/SFT_tools/Cherry_LLM/cherry_seletion/data_by_IFD.py \
--pt_data_path "$pt_data_file" \
--json_save_path "$selected_data_path" \
--json_data_path "$unselected_data_path" \
--max_length 512 \
--prompt openai \
--mod cherry \
--sample_rate 0.3
EOT

    # Submit the job
    sbatch "$sbatch_script"
    
    # Remove the temporary script
    rm "$sbatch_script"
    
    # Sleep for 2 seconds
    sleep 2
done