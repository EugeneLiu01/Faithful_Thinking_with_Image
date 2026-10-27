from pathlib import Path
import sys

eval_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(eval_root))

from datasets import interleave_datasets, load_dataset, load_from_disk
import os
import argparse
from qwen_vl_utils import process_vision_info
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from transformers.image_utils import load_image
from vllm import LLM, SamplingParams
import json
import copy
from tqdm import tqdm
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist

from PIL import Image
import torch
import numpy as np
from qwen_vl_utils import process_vision_info, smart_resize
import pickle

from common.compute_reward import rule_reward
from intervention.utils_intervention_v2 import do_intervention
from common.utils import setup_ddp, cleanup_ddp, merge_results
from data.utils import resolve_image_paths

MIN_PIXELS=401408     # 401408 
MAX_PIXELS=4014080 # define the image resolution
IMAGE_FACTOR = 28

# Resolve bundled paths relative to this script, independent of the working directory.
project_root = eval_root.parents[1]
data_root = eval_root / "data"
all_dp = {
    "vstar": str(data_root / "processed" / "vstar_bench_format.parquet"),
}


parser = argparse.ArgumentParser()
parser.add_argument('--model_base', type=str, default=str(project_root / "models" / "PixelReasoner-RL-v1"), help='Path to the model')
parser.add_argument('--model_name', type=str, default='qwen', help='Model name for result save')
parser.add_argument('--benchmark', type=str, default="vstar", choices=list(all_dp), help='Specify a benchmark')
parser.add_argument('--save_path', type=str, default=None, help='Path to save the results')
parser.add_argument('--num_workers', type=int, default=8)
args = parser.parse_args()

def to_rgb(pil_image: Image.Image) -> Image.Image:
      if pil_image.mode == 'RGBA':
          white_background = Image.new("RGB", pil_image.size, (255, 255, 255))
          white_background.paste(pil_image, mask=pil_image.split()[3])  # Use alpha channel as mask
          return white_background
      else:
          return pil_image.convert("RGB")

def resize_cropped(image, min_pixels=None, max_pixels=None):
    image = to_rgb(image)
    width, height = image.size
    if min_pixels is None: min_pixels =  MIN_PIXELS
    if max_pixels is None: max_pixels = MAX_PIXELS
    resized_height, resized_width = smart_resize(
        height,
        width,
        factor=IMAGE_FACTOR,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    print('resize cropped min max', min_pixels, max_pixels)
    image = image.resize((resized_width, resized_height))
    return image 

# ZOOMIN padding rate
ZOOMIN_PADDING_RATE = float(os.environ.get("ZOOMIN_PADDING_RATE", 0.1))

def zoom(raw_image, image, bbox_2d,  padding=ZOOMIN_PADDING_RATE):
    """
    Crop the image based on the bounding box coordinates.
    """
    img_x, img_y = image.size
    if bbox_2d[0] < 1 and bbox_2d[1] < 1 and bbox_2d[2] < 1 and bbox_2d[3] < 1:
        raise ValueError(f"Bounding box coordinates should be absolute pixel coordinates.")
    else:
        normalized_bbox_2d = (float(bbox_2d[0])/img_x-padding, float(bbox_2d[1])/img_y-padding, float(bbox_2d[2])/img_x+padding, float(bbox_2d[3])/img_y+padding)
    normalized_x1, normalized_y1, normalized_x2, normalized_y2 = normalized_bbox_2d
    normalized_x1 =min(max(0, normalized_x1), 1)
    normalized_y1 =min(max(0, normalized_y1), 1)
    normalized_x2 =min(max(0, normalized_x2), 1)
    normalized_y2 =min(max(0, normalized_y2), 1)
    raw_img_x, raw_img_y = raw_image.size
    cropped_img = raw_image.crop((normalized_x1*raw_img_x, normalized_y1*raw_img_y, normalized_x2*raw_img_x, normalized_y2*raw_img_y))
    w, h = cropped_img.size
    assert w > 28 and h > 28, f"Cropped image is too small: {w}x{h}"
    return cropped_img  

do_controlled_rectify = True
def execute_tool(images, rawimages, args, toolname, is_video, function=None):
    if toolname=='select_frames':
        tgt = args['target_frames']
        if len(tgt)>8:
            assert False, f"Execution Error: You have selected {len(tgt)} frames in total. Think again which frames you need to check in details (no more than 8 frames)"
            message = f"You have selected {len(tgt)} frames in total. Think again which frames you need to check in details (no more than 8 frames)"
            ##### controlled modification
            if do_controlled_rectify and np.random.uniform()<0.75:
                if np.random.uniform()<0.25:
                    tgt = tgt[:len(tgt)//2]
                elif np.random.uniform()<0.25/0.75:
                    tgt = tgt[-len(tgt)//2:]
                elif np.random.uniform()<0.25/0.5:
                    tgt = tgt[::2]
                else:
                    tgt = np.random.choice(tgt, size=len(tgt)//2, replace=False)
                    tgt = sorted(tgt)
                selected_frames = function(images[0], tgt)
                message = tgt
            else: 
                selected_frames = []
        elif max(tgt)>len(images[0]):
            message = f"There are {len(images[0])} frames numbered in range [1,{len(images[0])}]. Your selection is out of range."
            selected_frames = []
        else:
            message = ""
            candidates = images[0]
            if not isinstance(candidates, list):
                candidates = [candidates]
            selected_frames = function(candidates, [x-1 for x in tgt]) # video is always in the first item
        return selected_frames, message
    else:
        tgt = args['target_image']
        if is_video:
            if len(rawimages)==1: # there is only 
                # we default the candidate images into video frames 
                raw_video_frames = rawimages[0]
                video_frames = images[0]
                index = tgt - 1 
                assert index<len(video_frames), f"Incorrect `target_image`. You can only select frames in the given video within [1,{len(video_frames)}]"
                rawimage_to_crop = raw_video_frames[index]
                image_to_crop = video_frames[index]
            else: # there are zoomed images after the video; images = [[video], img, img, img]
                raw_cand_images = rawimages #[1:]
                cand_images = images    #[1:]
                index = tgt -1
                assert index<len(cand_images), f"Incorrect `target_image`. You can only select a previous frame within [1,{len(cand_images)}]"
                rawimage_to_crop = raw_cand_images[index]
                image_to_crop = cand_images[index]
        else:
            index =  tgt-1 
            assert index<len(images), f"Incorrect `target_image`. You can only select previous images within [1,{len(images)}]"
            
            rawimage_to_crop = rawimages[index]
            image_to_crop = images[index]
        if function is None: function = zoom
        cropped_image = function(rawimage_to_crop, image_to_crop, args['bbox_2d'])
    return cropped_image

def is_toolcalled(chat_messages):
    for msg in chat_messages:
        if msg['role'] == 'user' and len(msg['content']) == 2:
            if msg['content'][0]['type'] == 'text' and \
            "\nHere is the cropped image (Image Size:" in msg['content'][0]['text']:
                return True
    return False

def parse_last_tool(output_text):
    return json.loads(output_text.split(tool_start)[-1].split(tool_end)[0])

def intervent_sentence_before_toolcall(img_path, output_text, question, answer):
    # 获取tool call 前面的文本 Ta
    text_before_toolcall = output_text.split(tool_start)[0].strip()
    # 获取tool call 及其后面的文本
    text_tocall_and_after = output_text.split(text_before_toolcall)[1]
    # 干预文本
    intervented_text = do_intervention(img_path, question, answer, text_before_toolcall)
    # 拼回去
    output_text = intervented_text + text_tocall_and_after
    return output_text

tool_end = '</tool_call>'
tool_start = '<tool_call>'

def model_inference(model, processor, input_dict):
    text = input_dict["text"] if "text" in input_dict else input_dict["question"]
    images = input_dict["image"]
    answer = input_dict["answer"].split('\\boxed{')[-1].split('}')[0]

    all_images = []
    current_message_images = []

    # prompt from repo
    sys1 = "You are a helpful assistant."
    sys2 = "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{\"type\": \"function\", \"function\": {\"name\": \"crop_image\", \"description\": \"Zoom in on the image based on the bounding box coordinates. It is useful when the object or text in the image is too small to be seen.\", \"parameters\": {\"type\": \"object\", \"properties\": {\"bbox_2d\": {\"type\": \"array\", \"description\": \"The bounding box of the region to zoom in, as [x1, y1, x2, y2], where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner.\", \"items\": {\"type\": \"number\"}}, \"target_image\": {\"type\": \"number\", \"description\": \"The index of the image to crop. Index from 1 to the number of images. Choose 1 to operate on original image.\"}}, \"required\": [\"bbox_2d\", \"target_image\"]}}}\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call>"
    
    messages = [{
        "role": "system",
        "content": [
            {"type": "text", "text": sys1},
            {"type": "text", "text": sys2},
        ]
    }]
    print_messages = [{
        "role": "system",
        "content": [
            {"type": "text", "text": sys1},
            {"type": "text", "text": sys2},
        ]
    }]

    hint = "\n\nGuidelines: Understand the given visual information and the user query. Determine if it is beneficial to employ the given visual operations (tools). For an image, we can look closer by `crop_image`. Reason with the visual information step by step, and put your final answer within \\boxed{}."
    trigger = f"\n\n{hint}"     
    
    rawimagelist = current_message_images = [load_image(image) for image in images]

    messages.append({
        "role": "user",
        "content": [
            *[{"type": "image", "image": image, "min_pixels": 256*28*28, "max_pixels": 8000*28*28} for image in current_message_images],
            {"type": "text", "text": text+trigger},
        ],
    })
    print_messages.append({
        "role": "user",
        "content": [
            *[{"type": "image", "image": str(image), "min_pixels": 256*28*28, "max_pixels": 8000*28*28} for image in current_message_images],
            {"type": "text", "text": text+trigger},
        ],
    })
    
    # add to all_images
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    all_images += image_inputs
    all_raw_images = rawimagelist

    status = 'success'
    complete_assistant_response_for_gradio = []
    try_count = 0
    saved_org_thought_text = []

    while True:
        """
        Generate and stream text
        """
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)

        mm_data = {}
        mm_kw = {}
        if image_inputs:
            mm_data['image'] = image_inputs
        if video_inputs:
            mm_data['video'] = video_inputs
            for k, v in video_kwargs.items():
                mm_kw[k] = v[0]
        llm_inputs = {
            'prompt': prompt,
            "multi_modal_data": mm_data,
            "multi_modal_kwargs": mm_kw,
        }


        sampling_params = SamplingParams(
            n=1,
            temperature=0.0,
            top_p=1,
            top_k=-1,
            max_tokens=2048,
            min_tokens=1,
            skip_special_tokens=False,
            include_stop_str_in_output=False,
            stop=['<|im_end|>','<|eot_id|>','<|endoftext|>'],
        )
        try:
            all_outputs = model.generate(llm_inputs, sampling_params=sampling_params, use_tqdm=False)
        except Exception as e:
            print(f"Error!!!!", e)
            status = 'error'
            break
        completion_ids = [output.token_ids for outputs in all_outputs for output in outputs.outputs]
        output_text = processor.batch_decode(completion_ids, skip_special_tokens=True)
        response_message = output_text[0]


        require_tool = response_message.endswith(tool_end)

        if require_tool:
            if try_count == 0:
                # 只扰动第一次tool call的内容
                saved_org_thought_text.append(response_message)
                response_message = intervent_sentence_before_toolcall(images[0], response_message, text, answer)
            
            assistant_piece = dict(role='assistant', content=[
                dict(type='text', text=response_message),
            ])
            messages.append(assistant_piece)
            print_messages.append({
                "role": "assistant",
                "content": [
                    {"type": "text", "text": response_message},
                ],
            })
            print(print_messages)  
            try:
                tool_params = parse_last_tool(response_message) 
                tool_name = tool_params['name']
                tool_args = tool_params['arguments']
                video_flag = False
                raw_result = execute_tool(all_images, all_raw_images, tool_args, tool_name, is_video=video_flag)
                proc_img = resize_cropped(raw_result)
                all_images += [proc_img]
                all_raw_images += [raw_result]

                tool_call_piece = dict(role='user', content=[
                                dict(type='text', text="\nHere is the cropped image (Image Size: {}x{}):".format(proc_img.size[0], proc_img.size[1])),
                                dict(type='image', image=proc_img)
                                ])
                messages.append(tool_call_piece)
                print_messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "\nHere is the cropped image (Image Size: {}x{}):".format(proc_img.size[0], proc_img.size[1])},
                        {"type": "image", "image": str(proc_img)},
                    ],
                })
            except Exception as e:
                print(f"Error!!!!", e)
                error_info = str(e)
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"\nExecution error:\n{error_info}\n"},
                    ],
                })
                print_messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"\nExecution error:\n{error_info}\n"},
                    ],
                })
        else:
            assistant_piece = dict(role='assistant', content=[
                dict(type='text', text=response_message),
            ])
            messages.append(assistant_piece)
            print_messages.append({
                "role": "assistant",
                "content": [
                    {"type": "text", "text": response_message},
                ],
            })
            print(print_messages)            
            break

        try_count += 1


    saved_info = {}
    saved_info['qid'] = input_dict['qid']
    saved_info['category'] = input_dict['category']
    saved_info['image'] = input_dict["image"]
    saved_info['question'] = input_dict["question"]
    saved_info['answer'] = input_dict["answer"]
    saved_info['all_images'] = all_images
    saved_info['solution'] = response_message
    saved_info['messages'] = print_messages
    saved_info['saved_org_thought_text'] = saved_org_thought_text
    saved_info['status'] = status

    return saved_info, messages

def setup_vllm(model_base, limit_image_per_prompt=64, limit_video_per_prompt=8):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)
    print("Setting up VLLM model on rank:", local_rank)
    from vllm import LLM, SamplingParams
    model = LLM(
        model=model_base,
        revision=None,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.8,
        dtype="bfloat16",
        enable_prefix_caching=False,
        max_model_len=20480,
        distributed_executor_backend="external_launcher",
        seed=42,
        limit_mm_per_prompt={"image": limit_image_per_prompt, "video": limit_video_per_prompt},
    )
    processor = AutoProcessor.from_pretrained(model_base)
    return model, processor

if __name__ == "__main__":

    # saved path
    exp_path = args.save_path
    os.makedirs(exp_path, exist_ok=True)
    # save the args in exp_path
    with open(os.path.join(exp_path, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    setup_ddp()
    model_base = args.model_base

    model, processor = setup_vllm(model_base)
    tokenizer = processor.tokenizer

    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = dist.get_world_size()

    # load dataset
    dp = all_dp[args.benchmark]
    eval_data = load_dataset("parquet", data_files=dp)['train']
    eval_data = eval_data.map(resolve_image_paths, fn_kwargs={"data_dir": str(Path(dp).parent)})
    print(eval_data)
    eval_dataset = interleave_datasets(
        [eval_data],
        seed=42,
        stopping_strategy="first_exhausted",
    )
    sampler = DistributedSampler(eval_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    eval_dataloader = DataLoader(eval_dataset, batch_size=1, sampler=sampler, num_workers=4, collate_fn=lambda x: x)

    ret_results = []
    for data_batch in tqdm(eval_dataloader):
        data = data_batch[0]
        saved_info, messages = model_inference(model, processor, data)
        has_toolcall = is_toolcalled(messages); saved_info['has_toolcall'] = has_toolcall

        qid = saved_info['qid']
        save_path = os.path.join(exp_path, "samples", qid); os.makedirs(save_path, exist_ok=True)
        # save the messages
        with open(os.path.join(save_path, 'chat_message.pkl'), 'wb') as f:
            pickle.dump(messages, f)
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
    
    with open(os.path.join(exp_path, f'results_{rank}.json'), 'w') as f:
        json.dump(ret_results, f, indent=4)

    dist.barrier()
    if rank == 0:
        ret_results = merge_results(exp_path, 'results')

        error_cases = []
        failed_cases = []
        for result in ret_results:
            sol = result['solution']
            gt = result['answer']
            question = result['question']
            reward = 0

            if result['status'] != "success":
                failed_cases.append(result)
                reward = 0
            else:
                try:
                    reward = rule_reward(question, sol, gt, eostoken=tokenizer.eos_token, format_type=None, requires_box=True)[-1]
                except:
                    reward = 0

            result['acc'] = reward
            # Keep the accuracy score consumed by the McNemar test.
            result['score'] = result['acc']

            if result['score'] < 0.5:
                error_cases.append(result)
        
        # save the results
        with open(os.path.join(exp_path, 'results.json'), 'w') as f:
            json.dump(ret_results, f, indent=4)

        acc_results = {
            "acc": sum([i['acc'] for i in ret_results])/len(ret_results),
            "failed_case_num": len(failed_cases),
            "has_toolcall": sum([i['has_toolcall'] for i in ret_results]),
            "total_num": len(ret_results),
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
                    "has_toolcall": sum([i['has_toolcall'] for i in category_cases]),
                    "total_num": len(category_cases),
                    "error_cases": [i for i in category_cases if i['score'] < 0.5],
                    "failed_cases": [i for i in category_cases if i['status'] != "success"],
                }

                with open(os.path.join(exp_path, f'acc_results_{category}.json'), 'w') as f:
                    json.dump(category_results, f, indent=4)
                
                with open(os.path.join(exp_path, f'results_{category}.json'), 'w') as f:
                    json.dump(category_cases, f, indent=4)

        ################## process tool cases ##################
        tool_save_path = os.path.join(exp_path, "has_toolcall")
        os.makedirs(tool_save_path, exist_ok=True)
        tool_ret_results = [ret_results[i] for i in range(len(ret_results)) if ret_results[i]["has_toolcall"]]
        tool_error_cases = [error_cases[i] for i in range(len(error_cases)) if error_cases[i]["has_toolcall"]]
        with open(os.path.join(tool_save_path, 'results.json'), 'w') as f:
            json.dump(tool_ret_results, f, indent=4)
        
        tool_acc_results = {
            "acc": sum([i['acc'] for i in tool_ret_results])/len(tool_ret_results),
            "has_toolcall": sum([i['has_toolcall'] for i in tool_ret_results]),
            "total_num": len(tool_ret_results),
            "error_cases": tool_error_cases,
        }
        with open(os.path.join(tool_save_path, 'acc_results.json'), 'w') as f:
            json.dump(tool_acc_results, f, indent=4)
        
        if 'vstar' in args.benchmark:
            categories = ['direct_attributes', 'relative_position']
            for category in categories:
                category_cases = [i for i in tool_ret_results if i['category'] == category]
                category_acc = sum([i['acc'] for i in category_cases])/len(category_cases)
                category_results = {
                    "acc": category_acc,
                    "has_toolcall": sum([i['has_toolcall'] for i in category_cases]),
                    "total_num": len(category_cases),
                    "error_cases": [i for i in category_cases if i['score'] < 0.5],
                }

                with open(os.path.join(tool_save_path, f'acc_results_{category}.json'), 'w') as f:
                    json.dump(category_results, f, indent=4)
                
                with open(os.path.join(tool_save_path, f'results_{category}.json'), 'w') as f:
                    json.dump(category_cases, f, indent=4)

    
    cleanup_ddp()
    print("Inference done.")
