#!/usr/bin/env bash

benchmark=vstar
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
eval_path="$(cd -- "${script_dir}/.." && pwd)"

# zoom-in path to evaluate
zoomin_path="$1"
savefolder="$2"

# timestamp=$(date +"%Y%m%d_%H%M%S")
tagname="eval_${benchmark}_Faithfulness"

n_votes=5
vote_type="majority"

# record the zoom-in path
mkdir -p "${savefolder}/${tagname}"
echo "zoomin_path: ${zoomin_path}" > "${savefolder}/${tagname}/zoomin_path.txt"
echo "n_votes: ${n_votes}" >> "${savefolder}/${tagname}/zoomin_path.txt"
echo "vote_type: ${vote_type}" >> "${savefolder}/${tagname}/zoomin_path.txt"

##########################################################

## check relevance and sufficiency

# judger model

# Use the same GPT-4o environment configuration as doT.
api_url="${AZURE_OPENAI_ENDPOINT:-your-openai-endpoint}"
api_version="${OPENAI_API_VERSION:-2025-04-16}"
api_key="${AZURE_OPENAI_API_KEY:-your-api-key}"
eval_model_name="${AZURE_OPENAI_DEPLOYMENT:-gpt-4o-2024-11-20}"

# sufficiency - image: answer by zoom-in image
python "${eval_path}/visual_faithfulness/answer_by_zoomin_oai.py" \
    --zoomin_path "${zoomin_path}" \
    --eval_model_name "${eval_model_name}" \
    --api_url "${api_url}" \
    --api_key "${api_key}" \
    --api_version "${api_version}" \
    --benchmark "${benchmark}" \
    --n_votes "${n_votes}" \
    --vote_type "${vote_type}" \
    --save_path "${savefolder}/${tagname}/answer_by_zoomin"

# relevance - image: pred_ans rely on zoom-in image
python "${eval_path}/visual_faithfulness/ans_zoomin_relation_oai.py" \
    --zoomin_path "${zoomin_path}" \
    --eval_model_name "${eval_model_name}" \
    --api_url "${api_url}" \
    --api_key "${api_key}" \
    --api_version "${api_version}" \
    --benchmark "${benchmark}" \
    --n_votes "${n_votes}" \
    --vote_type "${vote_type}" \
    --save_path "${savefolder}/${tagname}/ans_zoomin_relation"
