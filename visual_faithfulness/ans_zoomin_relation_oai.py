from pathlib import Path
import sys

eval_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(eval_root))

from datasets import interleave_datasets, load_dataset, load_from_disk
import os
import argparse
from data.utils import resolve_image_paths
# from qwen_vl_utils import process_vision_info
# from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
# from transformers.image_utils import load_image
# from vllm import LLM, SamplingParams
import json
import copy
from tqdm import tqdm
# from torch.utils.data import DataLoader, DistributedSampler

from PIL import Image
from io import BytesIO
import base64
import torch
import numpy as np
# from qwen_vl_utils import process_vision_info, smart_resize
import pickle
from openai import OpenAI, AzureOpenAI
import requests
import multiprocessing
import time
import ast
multiprocessing.set_start_method('spawn', force=True)

MIN_PIXELS=401408
MAX_PIXELS=4014080 # define the image resolution
IMAGE_FACTOR = 28

data_root = eval_root / "data"
all_dp = {
    "vstar": str(data_root / "processed" / "vstar_bench_format.parquet"),
}

parser = argparse.ArgumentParser()
parser.add_argument('--zoomin_path', type=str, default=None, help='Path to save the results')
parser.add_argument('--eval_model_name', type=str, default=None, help='Model name for evaluation')
parser.add_argument('--api_key', type=str, default='EMPTY', help='API key')
parser.add_argument('--api_url', type=str, default='http://127.0.0.1:12348/v1', help='API URL')
parser.add_argument('--api_version', type=str, default='2025-04-16', help='API version')
parser.add_argument('--benchmark', type=str, default="vstar", choices=list(all_dp), help='Specify a benchmark')
parser.add_argument('--save_path', type=str, default=None, help='Path to save the results')
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--n_votes', type=int, default=5, help='Number of votes')
parser.add_argument('--vote_type', type=str, default='majority', choices=['majority', 'average', 'max'], help='Vote type')
args = parser.parse_args()

assert args.zoomin_path is not None


openai_api_key = args.api_key
openai_api_base = args.api_url
api_version = args.api_version

client = AzureOpenAI(
    azure_endpoint=openai_api_base,
    api_version=api_version,
    api_key=openai_api_key,
)

if args.eval_model_name is None:
    eval_model_name = "o3-2025-04-16"
else:
    eval_model_name = args.eval_model_name

nvotes = args.n_votes
vote_type = args.vote_type

instruction_prompt_system = """You are an image evidence validator.
Determine if the provided image regions support the entire answer statement.

**Input:**
- Question: [question about larger image]
- Image regions: One or more crops with bbox [left, top, right, bottom]
- Answer: [textual statement to validate]

**Validation Process:**
1. CAREFULLY EXAMINE THE CONTENT OF ALL PROVIDED IMAGE REGIONS
2. Extract ALL key claims from the Answer
3. For each claim:
   - Check if it's directly visible in any region
   - Or can be logically inferred from visible elements
4. If ALL claims are supported → "Yes"
5. If ANY claim lacks support → "No"

**Critical Rules:**
- Base decisions ONLY on what is visible in the provided regions
- Never use external knowledge (ignore words like "typically" or "usually")
- Reject if reasoning requires information beyond what's shown

**Examples:**
[Positive] 
Regions show: chopped vegetables
Answer: Food preparation in progress
→ "Yes" (core action visible)

[Negative]
Regions show: clothes on floor
Answer: Person changed clothes for work
→ "No" ("work" and "morning" not visible)

**Output:**
"Yes" or "No" (single word only)
"""

# saved path
exp_path = os.path.join(args.save_path, eval_model_name)
os.makedirs(exp_path, exist_ok=True)

def is_toolcalled(chat_messages):
    for msg in chat_messages:
        if msg['role'] == 'user' and len(msg['content']) == 2:
            if msg['content'][0]['type'] == 'text' and \
            "\nHere is the cropped image (Image Size:" in msg['content'][0]['text']:
                return True
    return False

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


tool_end = '</tool_call>'
tool_start = '<tool_call>'
def get_zoomin_imgs_bboxes(chat_messages):
    def parse_last_tool(output_text):
        # print([output_text])
        return json.loads(output_text.split(tool_start)[-1].split(tool_end)[0])

    all_images = []
    zoomin_imgs_bboxes = []
    for idx in range(len(chat_messages)):
        msg = chat_messages[idx]
        if msg['role'] == 'user' and len(msg['content']) == 2:
            if msg['content'][0]['type'] == 'image' and idx == 1:
                all_images.append(msg['content'][0]['image'])

            elif msg['content'][0]['type'] == 'text' and "\nHere is the cropped image (Image Size:" in msg['content'][0]['text']:
                all_images.append(msg['content'][1]['image'])

                # 上条message里有tool call
                last_msg = chat_messages[idx-1]
                tool_params = parse_last_tool(last_msg['content'][0]['text'])
                bbox_2d = tool_params['arguments']['bbox_2d']
                bbox_2d = {'bbox_2d': bbox_2d}

                zoomin_imgs_bboxes.append((msg['content'][1]['image'], bbox_2d))
    return all_images, zoomin_imgs_bboxes

def get_answer_state(chat_messages):
    ans_mgs = chat_messages[-1]
    ans_state = ans_mgs['content'][0]['text']
    ans_state = ans_state.split("\\boxed{")[-1].split("}")[0]
    return ans_state

def encode_pil_image_to_base64(pil_image):
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str

def encode_pil_image_to_base64(pil_image):
    buffered = BytesIO()
    pil_image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str

def model_inference(data):
    input_dict, chat_messages = data

    all_images, zoomin_imgs_bboxes = get_zoomin_imgs_bboxes(chat_messages)
    ans_states = get_answer_state(chat_messages)

    text = input_dict["text"] if "text" in input_dict else input_dict["question"]
    # 去掉可能带来干扰的后缀
    suffix_to_remove = "Answer with the option's letter from the given choices directly."
    if text.endswith(suffix_to_remove):
        text = text[:-len(suffix_to_remove)]
    images = input_dict["image"]
    qid = input_dict["qid"]

    # 检查是否已经处理过该qid
    is_processed = False
    saved_info_path = os.path.join(exp_path, "samples", qid, 'info.json')
    if os.path.exists(saved_info_path):
        # 已经处理过，读取直接返回
        is_processed = True
        with open(saved_info_path, 'r') as f:
            saved_info = json.load(f)
            return saved_info, is_processed

    # system prompt
    messages = [{
        "role": "system",
        "content": instruction_prompt_system,
    }]
    print_messages = [{
        "role": "system",
        "content": instruction_prompt_system,
    }]

    user_input_contents = []
    print_user_input_contents = []

    # question part
    user_input_contents.extend([
        {"type": "text", "text": text},
        {"type": "text", "text": "Input Images:"}
    ])
    print_user_input_contents.extend([
        {"type": "text", "text": text},
        {"type": "text", "text": "Input Images:"}
    ])

    # input image part
    for img, bbox in zoomin_imgs_bboxes:
        base64_img = encode_pil_image_to_base64(img)
        # print(str(bbox))
        user_input_contents.extend([
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}},
            {"type": "text", "text": f"{str(bbox)}"},
        ])
        print_user_input_contents.extend([
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64"}},
            {"type": "text", "text": f"{str(bbox)}"},
        ])

    # answer part
    user_input_contents.extend([
        {"type": "text", "text": "Answer:\n" + ans_states},
    ])
    print_user_input_contents.extend([
        {"type": "text", "text": "Answer:\n" + ans_states},
    ])


    messages.extend([
        {"role": "user", "content": user_input_contents},
    ])
    print_messages.extend([
        {"role": "user", "content": print_user_input_contents},
    ])
    # print(messages)
    
    response_message = ""
    status = 'success'

    votes_print_msgs = []
    votes_msgs = []
    votes_ans = []

    for i in range(nvotes):
        cur_msg = messages.copy()
        cur_print_msg = print_messages.copy()

        response_message = ""
        while not response_message:
            try:
                completion = client.chat.completions.create(
                    model=eval_model_name,
                    messages=cur_msg,
                    max_tokens=1024,
                    extra_headers={"X-TT-LOGID": ""},
                    temperature=0.0,
                )
                # break   # Exit the loop if the request is successful
                response_message = completion.choices[0].message.content
                if not response_message: print("Empty response message! Try again!!!"); time.sleep(10)
            except Exception as e:
                print(f"Error!!!! {qid} {e}")
                # 解析出error msg
                error_msg = str(e).split(" - ")[1].strip()
                error_json = ast.literal_eval(error_msg)
                print(f"Error json {error_json}")
                if error_json['error']['code'] == '-4003':
                    print(f"Image size exceeds limit, skip {qid}")
                    break
                time.sleep(10)


        output_text = response_message
        cur_msg.append({
            "role": "assistant",
            "content": response_message,
        })
        cur_print_msg.append({
            "role": "assistant",
            "content": response_message,
        })

        votes_msgs.append(cur_msg)
        votes_print_msgs.append(cur_print_msg)
        votes_ans.append(output_text)

    saved_info = {}
    saved_info['qid'] = input_dict['qid']
    saved_info['category'] = input_dict['category']
    saved_info['image'] = input_dict["image"]
    saved_info['question'] = input_dict["question"]
    saved_info['answer'] = input_dict["answer"]
    saved_info['all_images'] = all_images
    saved_info['solution'] = votes_ans
    saved_info['messages'] = votes_print_msgs
    saved_info['status'] = status

    return saved_info, is_processed

if __name__ == "__main__":
    pool = multiprocessing.Pool(processes=args.num_workers)
    
    # load dataset
    dp = all_dp[args.benchmark]
    eval_data = load_dataset("parquet", data_files=dp)['train']
    eval_data = eval_data.map(resolve_image_paths, fn_kwargs={"data_dir": str(Path(dp).parent)})
    eval_dataset = interleave_datasets(
        [eval_data],
        seed=42,
        stopping_strategy="first_exhausted",
    )
    print(len(eval_data))


    # 找到所有需要zoomin的样本
    toolcalled_data = []
    for data in tqdm(eval_dataset):
        qid = data['qid']
        sample_path = os.path.join(args.zoomin_path, "samples", qid)
        chat_messages = pickle.load(open(os.path.join(sample_path, "chat_message.pkl"), "rb"))
        if is_toolcalled(chat_messages):
            toolcalled_data.append((data, chat_messages))
    
    
    ret_results = []
    with tqdm(total=len(toolcalled_data), desc="Processing V*") as pbar:
        for saved_info, is_processed in pool.imap(model_inference, toolcalled_data):
            pbar.update(1)
            if not is_processed:
                qid = saved_info['qid']
                save_path = os.path.join(exp_path, "samples", qid); os.makedirs(save_path, exist_ok=True)
                # save the imgs
                all_images = saved_info['all_images']
                for i, img in enumerate(all_images):
                    img.save(os.path.join(save_path, f'{i}.png'))
                # save info
                del saved_info['all_images']
                with open(os.path.join(save_path, 'info.json'), 'w') as f:
                    json.dump(saved_info, f, indent=4)
            # add to ret_results
            ret_results.append(saved_info)

    pool.close()
    pool.join()
    
    with open(os.path.join(exp_path, f'results.json'), 'w') as f:
        json.dump(ret_results, f, indent=4)

    
    has_relation = 0
    for result in ret_results:
        has_relation_this = 0
        # sol = result['solution']
        for sol in result['solution']:
            if sol.lower() == "yes" or "yes" in sol.lower():
                has_relation_this += 1
            elif sol.lower() == "no" or "no" in sol.lower():
                pass
            else:
                print(f"Unknown vote: {sol}")
        if vote_type == 'majority':
            if has_relation_this > nvotes // 2:
                has_relation += 1
                result['has_relation'] = True
            else:
                result['has_relation'] = False
        elif vote_type == 'average':
            relation_rate = has_relation_this / nvotes
            has_relation += relation_rate
            result['has_relation'] = relation_rate
        elif vote_type == 'max':
            if has_relation_this > 0:
                has_relation += 1
                result['has_relation'] = True
            else:
                result['has_relation'] = False

    print(f"Has relation: {has_relation}/{len(ret_results)}")
    judge_result = {
        "has_relation": has_relation,
        "is_tool_call": len(ret_results),
        "total": len(eval_data),
        "tool_call_rate": len(ret_results) / len(eval_data),
        "relation_rate": has_relation / len(ret_results),
    }
    with open(os.path.join(exp_path, f'judge_result.json'), 'w') as f:
        json.dump(judge_result, f, indent=4)

    if 'vstar' in args.benchmark:
        categories = ['direct_attributes', 'relative_position']
        for category in categories:
            category_cases = [i for i in ret_results if i['category'] == category]
            category_has_relation = []
            category_no_relation = []
            for case in category_cases:
                if case['has_relation']:
                    category_has_relation.append(case)
                else:
                    category_no_relation.append(case)
            category_results = {
                "has_relation": len(category_has_relation),
                "total_num": len(category_cases),
                "relation_rate": len(category_has_relation) / len(category_cases),
                "cases_has_relation": category_has_relation,
                "cases_no_relation": category_no_relation,
                "failed_cases": [i for i in category_cases if i['status'] != "success"],
            }

            with open(os.path.join(exp_path, f'judge_result_{category}.json'), 'w') as f:
                json.dump(category_results, f, indent=4)
            
            with open(os.path.join(exp_path, f'results_{category}.json'), 'w') as f:
                json.dump(category_cases, f, indent=4)
