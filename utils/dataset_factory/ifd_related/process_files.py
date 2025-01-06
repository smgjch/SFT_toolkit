import os
import json
from typing import List
from tqdm import tqdm

error_list = []

def extract_data_from_json_files(file_paths):
    # 存储所有文件的_id的列表
    id_list = []
    # 存储所有文件的last_user_content的列表
    user_content_list = []
    ast_content_list = []

    id_start = 0
    # 遍历每个文件路径
    for file_path in file_paths:
        print(f'Processing file: {file_path}')
        # 检查文件是否存在并且是.json文件
        if os.path.isfile(file_path) and file_path.endswith('.json'):
            # 打开并读取JSON文件
            with open(file_path, 'r', encoding='utf-8') as file:
                # 如果是一个JSON列表
                data_list = json.load(file)
            json_with_ids = []
            # 如果是单个对象，将其转换为列表
            if not isinstance(data_list, list):
                data_list = [data_list]

            # 遍历每个数据对象
            for item in tqdm(data_list):

                item["_id"] = id_start
                id_start += 1
                json_with_ids.append(item)

                last_user_content = ''
                last_ast_content = ''
                # 从messages中找最后一个user和assistant的对话
                for msg_idx in range(len(item['messages'])-1):
                    if item['messages'][msg_idx]['role'] == 'user' and item['messages'][msg_idx+1]['role'] == 'assistant':
                        last_user_content = item['messages'][msg_idx]['content']
                        last_ast_content = item['messages'][msg_idx+1]['content']
                if len(last_user_content) > 0 and len(last_ast_content) > 0:
                    id_list.append(item['_id'])
                    user_content_list.append(last_user_content)
                    ast_content_list.append(last_ast_content)
                else:
                    error_list.append({'file_path': file_path, 'item': item})

            with open(file_path,"w", encoding='utf-8') as file:
                json.dump(json_with_ids, file, indent=4, ensure_ascii=False)

    return id_list, user_content_list, ast_content_list

def find_json_files(directory: str) -> List[str]:
    json_file_paths = []

    # 遍历目录
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.json'):
            # if file.endswith('.json'):
                # 获取绝对路径并添加到列表
                json_file_paths.append(os.path.abspath(os.path.join(root, file)))

    return json_file_paths

json_file_paths = find_json_files('/mnt/petrelfs/hujucheng/train/data/100Wraw')
# json_file_paths = find_json_files('/mnt/hwfile/opendatalab/panzhuoshi/data/sft_data_s1_0823_gpt4o_s1_hu_part2_id_meta')
total_ids, total_user_contents, total_ast_contents = extract_data_from_json_files(json_file_paths)

assert len(total_ids) == len(total_user_contents) == len(total_ast_contents)

# data = [{'_id': _id, 'instruction': user_content, 'output': ast_content} for _id, user_content, ast_content in zip(total_ids, total_user_contents, total_ast_contents)]
data = []
for _id, user_content, ast_content in zip(total_ids, total_user_contents, total_ast_contents):
    data.append({'_id': _id, 'instruction': user_content, 'output': ast_content})

# # write data to json file
# output_dir = '/mnt/petrelfs/peiqizhi/data/sft1v2_hu_zhuoshi_tmp.json'
# with open(output_dir, 'w', encoding='utf-8') as f:
#     json.dump(data, f, ensure_ascii=False, indent=4)
# # save the error list to json
# with open('/mnt/petrelfs/peiqizhi/data/sft1v2_hu_zhuoshi_error_list.json', 'w', encoding='utf-8') as f:
#     json.dump(error_list, f, ensure_ascii=False, indent=4)
# print(len(data))
# print(len(error_list))
# print('done')
# exit()

output_dir = '/mnt/petrelfs/hujucheng/train/data/100Wbase'

os.makedirs(output_dir, exist_ok=True)

# save the error list to json
with open(os.path.join(output_dir, 'error_list.json'), 'w', encoding='utf-8') as f:
    json.dump(error_list, f, ensure_ascii=False, indent=4)

n = 64

# 计算每一份的大小
chunk_size = len(data) // n
for i in range(n):
    # 确定每一份的起始和结束索引
    start = i * chunk_size
    # 对最后一份特殊处理，确保包含所有剩余的数据
    end = (i + 1) * chunk_size if i < n - 1 else len(data)
    
    # 获取当前分割的数据
    chunk_data = data[start:end]
    
    # 将该份保存为 JSON 文件
    filename = os.path.join(output_dir, f'data_part_{i+1}.json')
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(chunk_data, f, ensure_ascii=False, indent=4)

print(f'Data split into {n} parts and saved to {output_dir}')
