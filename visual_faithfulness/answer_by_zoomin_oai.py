from pathlib import Path
import sys

eval_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(eval_root))

from datasets import interleave_datasets, load_dataset, load_from_disk
import os
import argparse
from data.utils import resolve_image_paths
import json
import copy
from tqdm import tqdm
# from torch.utils.data import DataLoader, DistributedSampler

from PIL import Image
from io import BytesIO
import base64
import pickle
from openai import OpenAI, AzureOpenAI
import requests
import multiprocessing
import time
import ast
multiprocessing.set_start_method('spawn', force=True)

from common.compute_reward import rule_reward

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
parser.add_argument('--eos_token', type=str, default='<|im_end|>', help='EOS marker passed to rule_reward (Qwen default)')
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

instruction_prompt_system = """You are a helpful visual assistant.

You will receive one or more image crops from a larger scene, each accompanied by a 2D bounding box (in the format [left, top, right, bottom]) denoting its position in the original image.

**Example Input**:
`{"bbox_2d": [10, 20, 100, 200]}`

**Your Instructions**:
1.  **Base Answers Solely on Input**: Use **only** the provided visual crops and their bounding boxes to answer the user's query. **Do not use any external knowledge or assumptions**.
2.  **Handle Uncertainty**: If the given visual information is insufficient to answer the query, respond with "I don't know".
3.  **Format Output**: Always place your final answer within `\\boxed{}`.
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

    text = input_dict["text"] if "text" in input_dict else input_dict["question"]
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

    user_input_contents.extend([
        {"type": "text", "text": text},
    ])
    print_user_input_contents.extend([
        {"type": "text", "text": text},
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

    rew = []
    error_cases = []
    failed_cases = []
    for result in ret_results:
        gt = result['answer']
        question = result['question']

        this_reward = []
        for sol in result['solution']:
            if result['status'] != "success":
                failed_cases.append(result)
                reward = 0
            else:
                try:
                    reward = rule_reward(question, sol, gt, eostoken=args.eos_token, format_type=None, requires_box=True)[-1]
                except:
                    reward = 0
            this_reward.append(reward)

        # 5. Corrected and distinct voting logic
        vote_reward = 0.0
        if vote_type == 'majority':
            num_correct = sum(1 for r in this_reward if r > 0.5)
            vote_reward = 1.0 if num_correct > len(this_reward) / 2 else 0.0
        elif vote_type == 'average':
            if this_reward:
                vote_reward = sum(this_reward) / len(this_reward)
        elif vote_type == 'max':
            vote_reward = max(this_reward)
        # add to list
        rew.append(vote_reward)

        result['acc'] = vote_reward
        
        if vote_reward < 0.5:
            error_cases.append(result)

    acc_results = {
        "acc": sum(rew)/len(rew),
        "failed_case_num": len(failed_cases),
        "error_cases": error_cases,
        "failed_cases": failed_cases,
    }

    with open(os.path.join(exp_path, 'acc_results.json'), 'w') as f:
        json.dump(acc_results, f, indent=4)

    if 'vstar' in args.benchmark:
        categories = ['direct_attributes', 'relative_position']
        for category in categories:
            category_cases = [i for i in ret_results if i['category'] == category]
            category_acc = sum([i['acc'] for i in category_cases])/len(category_cases)
            category_results = {
                "acc": category_acc,
                "total_num": len(category_cases),
                "error_cases": [i for i in category_cases if i['acc'] < 0.5],
                "failed_cases": [i for i in category_cases if i['status'] != "success"],
            }

            with open(os.path.join(exp_path, f'acc_results_{category}.json'), 'w') as f:
                json.dump(category_results, f, indent=4)

            with open(os.path.join(exp_path, f'results_{category}.json'), 'w') as f:
                json.dump(category_cases, f, indent=4)
