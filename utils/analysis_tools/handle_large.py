import json

# Function to process the JSON file

def process_large_json(input_file):
    categorized_data = {}

    with open(input_file, 'r') as f:
        # Load the entire JSON content as it is a single large array
        data = json.load(f)

        for item in data:
            # Skip "Code" and "Harmless" categories
            category = item['category']
            if category in ["Code", "Harmless"]:
                continue

            # Append item to the respective category
            if category not in categorized_data:
                categorized_data[category] = []
            
            categorized_data[category].append(item)

    # Write each category to a separate JSON file
    for category, items in categorized_data.items():
        with open(f'{category}.json', 'w') as f:
            json.dump(items, f, indent=4)

# Call the function with your input file
process_large_json('/mnt/petrelfs/hujucheng/huLLM/data/moss-003-sft-no-tools.json')