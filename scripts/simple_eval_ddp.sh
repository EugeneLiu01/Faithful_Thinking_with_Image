#!/usr/bin/env bash

benchmark=vstar
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
eval_path="$(cd -- "${script_dir}/.." && pwd)"
policy="$1"

timestamp=$(date +"%Y%m%d_%H%M%S")
model_name="SCCM-RL"
savefolder="eval_exp"

tagname="eval_${benchmark}"

# Single-node torchrun configuration.
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-19427}"

torchrun \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    "${eval_path}/intervention/simple_eval_vllm_ddp.py" \
    --model_base "${policy}" \
    --model_name "${model_name}" \
    --benchmark "${benchmark}" \
    --save_path "${eval_path}/${savefolder}/${tagname}/${timestamp}"
