import torch.distributed as dist
import os
import torch
import datetime
from contextlib import contextmanager
import time
import json
import glob

def json_loads(text: str) -> dict:
    text = text.strip('\n')
    if text.startswith('```') and text.endswith('\n```'):
        text = '\n'.join(text.split('\n')[1:-1])
    try:
        return json.loads(text)
    except json.decoder.JSONDecodeError as json_err:
        try:
            return json.loads(text + "}")
        except Exception:
            raise json_err

@contextmanager
def timer(hint=""):
    start = time.perf_counter()
    yield
    end = time.perf_counter()
    # rank0_debug(f"🕙 {hint} runtime: {end - start:.2f} s")


def rank0_print(*args):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(f"Rank {dist.get_rank()}: ", *args)
    else:
        print(*args)


def rank_print(*args):
    if dist.is_initialized():
        print(f"Rank {dist.get_rank()}: ", *args)
    else:
        print(*args)


def rank0_debug(*args):
    if os.getenv("DEBUG") != "true":
        return
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(f"Rank {dist.get_rank()}: ", *args)
    else:
        print(*args)


def rank_debug(*args):
    if os.getenv("DEBUG") != "true":
        return
    if dist.is_initialized():
        print(f"Rank {dist.get_rank()}: ", *args)
    else:
        print(*args)


def setup_ddp():
    """初始化分布式环境。"""
    # 使用环境变量获取多机多卡的信息
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    # 初始化进程组
    dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=datetime.timedelta(days=365))
    torch.cuda.set_device(local_rank)
    
    print(f"初始化进程: rank={rank}, local_rank={local_rank}, world_size={world_size}")


def cleanup_ddp():
    """销毁分布式进程组。"""
    dist.destroy_process_group()

def merge_results(save_path, file_prefix):
    """合并所有 rank 的结果文件，仅主进程负责执行。"""
    print("Merging result files...")
    result_files = glob.glob(f"{save_path}/{file_prefix}_*.json")
    
    # 确保所有文件都存在
    world_size = dist.get_world_size()
    expected_files = [f"{save_path}/{file_prefix}_{i}.json" for i in range(world_size)]
    missing_files = [f for f in expected_files if f not in result_files]
    
    if missing_files:
        print(f"警告：缺少以下结果文件: {missing_files}")

    # 读文件合并
    # # 数据集长度不能被 num_replicas 整除时，DistributedSampler 会通过复制部分数据让每个进程处理的数据量相同。最后数据合并的时候可能需要去重
    all_data = []
    for file in sorted(result_files):
        with open(file, "r") as f:
            data = json.load(f)
            all_data.extend(data)
    # 依据qid去重
    unique_data = {item['qid']: item for item in all_data}
    all_data = list(unique_data.values())
    # 写文件合并
    with open(f"{save_path}/{file_prefix}_merged.json", "w") as writer:
        json.dump(all_data, writer, indent=4)
    print("结果合并成功。")
    return all_data
