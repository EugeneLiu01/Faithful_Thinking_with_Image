#!/usr/bin/env bash

benchmark=vstar
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
eval_path="$(cd -- "${script_dir}/.." && pwd)"

# Shared GPT-4o configuration for doT and visual faithfulness evaluation.
# Existing environment values take precedence.
export AZURE_OPENAI_ENDPOINT="${AZURE_OPENAI_ENDPOINT:-your-openai-endpoint}"
export AZURE_OPENAI_API_KEY="${AZURE_OPENAI_API_KEY:-your-api-key}"
export OPENAI_API_VERSION="${OPENAI_API_VERSION:-2025-04-16}"
export AZURE_OPENAI_DEPLOYMENT="${AZURE_OPENAI_DEPLOYMENT:-gpt-4o-2024-11-20}"

policy="$1"
# get the model_name from policy
model_name="SCCM-RL"


timestamp=$(date +"%Y%m%d_%H%M%S")
# model_name="test"
savefolder=${eval_path}/eval_exp/${model_name}/${timestamp}
mkdir -p "${savefolder}"

tagname="eval_${benchmark}"
echo "[!!!!] eval ${benchmark} with ${policy}" > "${savefolder}/${tagname}_log.txt"

# Single-node torchrun configuration.
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-19427}"
torchrun_args=(
    --nproc_per_node="${NPROC_PER_NODE}"
    --nnodes=1
    --node_rank=0
    --master_addr="${MASTER_ADDR}"
    --master_port="${MASTER_PORT}"
)

# default, no intervention
torchrun "${torchrun_args[@]}" \
    "${eval_path}/intervention/simple_eval_vllm_ddp.py" \
    --model_base "${policy}" \
    --model_name "${model_name}" \
    --benchmark "${benchmark}" \
    --save_path "${savefolder}/${tagname}"

# eval faithful
bash "${eval_path}/scripts/eval_faith_all.sh" "${savefolder}/${tagname}" "${savefolder}" &

# intervene on V with random noise
torchrun "${torchrun_args[@]}" \
    "${eval_path}/intervention/simple_eval_vllm_ddp_doV_randnoise.py" \
    --model_base "${policy}" \
    --model_name "${model_name}" \
    --benchmark "${benchmark}" \
    --save_path "${savefolder}/${tagname}_doV_randnoise"

# intervene on V with a mislocalized bounding box
torchrun "${torchrun_args[@]}" \
    "${eval_path}/intervention/simple_eval_vllm_ddp_doV_misbbox.py" \
    --model_base "${policy}" \
    --model_name "${model_name}" \
    --benchmark "${benchmark}" \
    --save_path "${savefolder}/${tagname}_doV_misbbox"

# intervent on T
torchrun "${torchrun_args[@]}" \
    "${eval_path}/intervention/simple_eval_vllm_ddp_doT.py" \
    --model_base "${policy}" \
    --model_name "${model_name}" \
    --benchmark "${benchmark}" \
    --save_path "${savefolder}/${tagname}_doT"

# statistics test
test_results=${savefolder}/${tagname}_statest
mkdir -p "${test_results}"
## 1. T -> answer
echo "[!!!!] T -> answer" >> "${test_results}/T_answer.txt"
echo "==== direct attributs ====" >> "${test_results}/T_answer.txt"
python "${eval_path}/intervention/mcnemar_test.py" \
--group_a "${savefolder}/${tagname}/has_toolcall/results_direct_attributes.json" \
--group_b "${savefolder}/${tagname}_doT/has_toolcall/results_direct_attributes.json" \
>> "${test_results}/T_answer.txt"

echo "==== relative position ====" >> "${test_results}/T_answer.txt"
python "${eval_path}/intervention/mcnemar_test.py" \
--group_a "${savefolder}/${tagname}/has_toolcall/results_relative_position.json" \
--group_b "${savefolder}/${tagname}_doT/has_toolcall/results_relative_position.json" \
>> "${test_results}/T_answer.txt"

echo "===== overall ====" >> "${test_results}/T_answer.txt"
python "${eval_path}/intervention/mcnemar_test.py" \
--group_a "${savefolder}/${tagname}/has_toolcall/results.json" \
--group_b "${savefolder}/${tagname}_doT/has_toolcall/results.json" \
>> "${test_results}/T_answer.txt"


## 2. V (random noise) -> answer
echo "[!!!!] V (random noise) -> answer" >> "${test_results}/V_randnoise_answer.txt"
echo "==== direct attributs ====" >> "${test_results}/V_randnoise_answer.txt"
python "${eval_path}/intervention/mcnemar_test.py" \
--group_a "${savefolder}/${tagname}/has_toolcall/results_direct_attributes.json" \
--group_b "${savefolder}/${tagname}_doV_randnoise/has_toolcall/results_direct_attributes.json" \
>> "${test_results}/V_randnoise_answer.txt"

echo "==== relative position ====" >> "${test_results}/V_randnoise_answer.txt"
python "${eval_path}/intervention/mcnemar_test.py" \
--group_a "${savefolder}/${tagname}/has_toolcall/results_relative_position.json" \
--group_b "${savefolder}/${tagname}_doV_randnoise/has_toolcall/results_relative_position.json" \
>> "${test_results}/V_randnoise_answer.txt"

echo "===== overall ====" >> "${test_results}/V_randnoise_answer.txt"
python "${eval_path}/intervention/mcnemar_test.py" \
--group_a "${savefolder}/${tagname}/has_toolcall/results.json" \
--group_b "${savefolder}/${tagname}_doV_randnoise/has_toolcall/results.json" \
>> "${test_results}/V_randnoise_answer.txt"

## 3. V (mislocalized bounding box) -> answer
echo "[!!!!] V (misbbox) -> answer" >> "${test_results}/V_misbbox_answer.txt"
echo "==== direct attributs ====" >> "${test_results}/V_misbbox_answer.txt"
python "${eval_path}/intervention/mcnemar_test.py" \
--group_a "${savefolder}/${tagname}/has_toolcall/results_direct_attributes.json" \
--group_b "${savefolder}/${tagname}_doV_misbbox/has_toolcall/results_direct_attributes.json" \
>> "${test_results}/V_misbbox_answer.txt"

echo "==== relative position ====" >> "${test_results}/V_misbbox_answer.txt"
python "${eval_path}/intervention/mcnemar_test.py" \
--group_a "${savefolder}/${tagname}/has_toolcall/results_relative_position.json" \
--group_b "${savefolder}/${tagname}_doV_misbbox/has_toolcall/results_relative_position.json" \
>> "${test_results}/V_misbbox_answer.txt"

echo "===== overall ====" >> "${test_results}/V_misbbox_answer.txt"
python "${eval_path}/intervention/mcnemar_test.py" \
--group_a "${savefolder}/${tagname}/has_toolcall/results.json" \
--group_b "${savefolder}/${tagname}_doV_misbbox/has_toolcall/results.json" \
>> "${test_results}/V_misbbox_answer.txt"

wait
