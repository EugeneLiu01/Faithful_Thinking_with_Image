import os
import json
import argparse
from pathlib import Path

"""
Data preparation for V*Bench evaluation following DeepEyes.
"""

def main():
    data_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Prepare V*Bench evaluation data")
    parser.add_argument('--input-dir', type=Path, default=data_root / 'raw' / 'vstar_bench',
                        help='Root directory of the downloaded V*Bench data')
    parser.add_argument('--output-file', type=Path,
                        default=data_root / 'processed' / 'vstar_bench_format.parquet',
                        help='Output Parquet path')
    args = parser.parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_file = args.output_file.expanduser().resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset

    test_types = ['direct_attributes', 'relative_position']

    abc_map = {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E', 6: 'F'}
    instruction_prompt_before = """Question: {question}
Options: {options}"""

    all_data = []
    question_id = 0
    for test_type in test_types:
        test_path = os.path.join(input_dir, test_type)
        image_files = list(filter(lambda file: '.json' not in file, os.listdir(test_path)))
        image_args = [[img, test_path] for img in image_files]
        thistype_data = []
        for image in image_args:
            img_name = image[0]
            path = image[1]
            # 加载json文件
            with open(os.path.join(path, img_name.replace('.jpg', '.json')), 'r') as f:
                anno = json.load(f)
            question = anno['question']
            options = anno['options']

            option_str = "\n"
            for i in range(len(options)):
                option_str += abc_map[i + 1] + '. ' + options[i] + '\n'

            prompt = instruction_prompt_before.format(question=question, options=option_str.strip())
            # 提取需要的字段
            element = {}
            # image 字段
            img_path = os.path.join(path, img_name)
            element['image'] = [os.path.relpath(img_path, output_file.parent)]
            # category 字段
            element['category'] = test_type
            # qid 字段
            element['qid'] = f"{test_type}_{img_name.split('.')[0]}"
            # question_id 字段
            element['question_id'] = question_id
            question_id += 1
            # label 字段
            element['label'] = "A. " + options[0]
            # answer 字段
            element['answer'] = "\\boxed{" + element['label'] + "}"
            # question & text 字段
            element['question'] = prompt
            element['text'] = prompt
            thistype_data.append(element)
        all_data.extend(thistype_data)

    print(f"Total samples: {len(all_data)}")
    # 保存数据到parquet文件
    from datasets import Dataset

    # 从列表创建dataset
    dataset = Dataset.from_list(all_data)

    # 保存为parquet格式
    dataset.to_parquet(str(output_file))
    print(f"Data saved to {output_file}")

    # 验证读取
    read_dataset = load_dataset('parquet', data_files=str(output_file))
    print(f"Read verification succeeded: {len(read_dataset['train'])} samples")


if __name__ == "__main__":
    main()
