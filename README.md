# On the Faithfulness of Visual Thinking: Measurement and Enhancement

<p align="center">
  <a href="https://arxiv.org/abs/2510.23482">
    <img src="https://img.shields.io/badge/arXiv-Paper-red" alt="arXiv">
  </a>
  <a href="https://github.com/EugeneLiu01/Faithful_Thinking_with_Image">
    <img src="https://img.shields.io/badge/GitHub-Code-181717?logo=github" alt="GitHub">
  </a>
  <a href="https://huggingface.co/eugeneliu01/Faithful_Thinking_with_Image">
    <img src="https://img.shields.io/badge/🤗 huggingface-Model-yellow" alt="Hugging Face">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-Apache%202.0-blue" alt="License">
  </a>
</p>


> **Abstract** Recent large vision-language models (LVLMs) can generate vision-text multimodal chain-of-thought (MCoT) traces after reinforcement fine-tuning (RFT). However, we observe that the visual information in MCoT is often inaccurate even when answers are correct, and task accuracy remains nearly unchanged despite the visual information being corrupted. This indicates a lack of faithfulness in the vision part of MCoT reasoning.
> We attribute this to *the RL reward design in RFT, which solely incentivizes the format of interleaved vision-text cues*, encouraging the model to incorporate visual information into its text reasoning steps without considering its correctness. In this paper, we first probe MCoT faithfulness by measuring how much the prediction changes when its visual and textual thoughts are intervened. Surprisingly, the model's predictions remain nearly unchanged under visual intervention but change significantly under textual intervention, indicating that **the visual evidence is largely ignored**.
> To further analyze the visual information, we introduce an automated LVLM-based evaluation metric that quantifies the faithfulness of visual cues from two perspectives: relevance and sufficiency. Our evaluation reveals that the visual information in current MCoT traces is simultaneously irrelevant and insufficient.
> To address this issue, we propose Sufficient-Component Cause Model (SCCM) learning. This approach encourages the MCoT to generate sufficient yet minimal visual components that are **independently capable of leading to the correct answer**.
> The proposed SCCM is annotation-free and plug-and-play, compatible with various RFT for MCoT. Empirical results demonstrate that SCCM substantially improves the visual faithfulness across a suite of fine-grained perception and reasoning benchmarks.

<p align="center">
  <a href="assets/FailCases5h.png">
    <img src="assets/FailCases5h.png" alt="Examples of irrelevant and insufficient visual information" width="100%">
  </a>
</p>

## Installation

```bash
git clone https://github.com/EugeneLiu01/Faithful_Thinking_with_Image.git
cd Faithful_Thinking_with_Image

conda create -n faithful-mcot python=3.10 -y
conda activate faithful-mcot
```

The provided `requirements.txt` records the reference environment and targets CUDA 12.4. Install the matching PyTorch wheels first, followed by FlashAttention and the remaining dependencies:

```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124
pip install flash-attn==2.7.4.post1 --no-build-isolation
pip install -r requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cu124
```

For a different CUDA version, install compatible PyTorch and vLLM builds first and update the CUDA-specific pins in `requirements.txt`.

## Data Preparation

Download V*Bench from Hugging Face:

```bash
hf download craigwu/vstar_bench \
  --repo-type dataset \
  --local-dir data/raw/vstar_bench
```

Convert the annotations into the format consumed by the evaluators:

```bash
python data/recons_data.py
```

The command generates:

```text
data/processed/vstar_bench_format.parquet
```

Keep both the generated Parquet file and the raw image directory. Image paths inside the evaluation data are resolved against the original images. See [data/README.md](data/README.md) for details and custom input paths.

## Service Configuration

### Local Answer Judge

The reward module uses an OpenAI-compatible judge service that exposes a `/models` endpoint. For example, you can deploy and use `Qwen/Qwen2.5-32B-Instruct` with vLLM in a separate terminal:

```bash
vllm serve Qwen/Qwen2.5-32B-Instruct \
  --host 127.0.0.1 \
  --port 12347
```

Keep the judge service running and configure its base URL in the evaluation terminal:

```bash
export LLM_AS_A_JUDGE_BASE="http://127.0.0.1:12347/v1"
```

### Azure OpenAI

Textual intervention and automated visual-faithfulness evaluation use Azure OpenAI:

```bash
export AZURE_OPENAI_ENDPOINT="https://<resource>.openai.azure.com"
export AZURE_OPENAI_API_KEY="<your-api-key>"
export OPENAI_API_VERSION="2025-04-16"
export AZURE_OPENAI_DEPLOYMENT="<your-gpt-4o-deployment>"
```

The current implementation uses the `AzureOpenAI` client. To use another provider, adapt the client initialization in `intervention/utils_intervention_v2.py` and `visual_faithfulness/*_oai.py`.

## Usage

### Full Evaluation

The scripts use `torchrun` in single-node mode and default to one GPU. Override the following standard variables when needed:

```bash
export NPROC_PER_NODE=1
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=19427
```

Download the released [SCCM-RL model](https://huggingface.co/eugeneliu01/Faithful_Thinking_with_Image), then run the complete pipeline with the local checkpoint path:

```bash
bash scripts/one4all.sh /path/to/SCCM-RL
```

The script launches the following workflow:

1. Baseline MCoT inference.
2. Automated relevance and sufficiency evaluation in the background.
3. Visual intervention with random-noise observations.
4. Visual intervention with mislocalized bounding boxes.
5. Textual-thought intervention.
6. Paired McNemar tests against the baseline.

The full pipeline writes results under:

```text
eval_exp/SCCM-RL/<timestamp>/
|-- eval_vstar/                         # Baseline
|-- eval_vstar_doV_randnoise/           # Random-noise intervention
|-- eval_vstar_doV_misbbox/             # Mislocalized-box intervention
|-- eval_vstar_doT/                     # Textual intervention
|-- eval_vstar_Faithfulness/
|   |-- answer_by_zoomin/               # Sufficiency
|   `-- ans_zoomin_relation/            # Relevance
`-- eval_vstar_statest/
    |-- T_answer.txt
    |-- V_answer.txt
    `-- V_misbbox_answer.txt
```

Each inference directory contains per-sample traces, merged predictions, accuracy summaries.



### Baseline Only

To run only the baseline inference:

```bash
bash scripts/simple_eval_ddp.sh /path/to/SCCM-RL
```

### Faithfulness Evaluation Only

Given an existing baseline output directory:

```bash
bash scripts/eval_faith_all.sh \
  /path/to/eval_vstar \
  /path/to/output_root
```

By default, the evaluator uses five votes with majority voting.

## Citation

If you find this work useful, please cite:

```bibtex
@article{liu2025faithfulness,
  title={On the Faithfulness of Visual Thinking: Measurement and Enhancement},
  author={Liu, Zujing and Pan, Junwen and She, Qi and Gao, Yuan and Xia, Guisong},
  journal={arXiv preprint arXiv:2510.23482},
  year={2025}
}
```

## Acknowledgements

We sincerely thank the following projects:
- [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL)
- [DeepEyes](https://github.com/Visual-Agent/DeepEyes)
- [Pixel-Reasoner](https://github.com/TIGER-AI-Lab/Pixel-Reasoner)
