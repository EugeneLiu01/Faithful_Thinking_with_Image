# V*Bench Data Preparation

Follow these steps to prepare the data required by `--benchmark vstar`. No GPU is needed for data preparation.

V*Bench data construction follows the approach used in [DeepEyes](https://github.com/Visual-Agent/DeepEyes).

Run all commands from `code_publish/eval_sccm_model/`. Skip the following command if you are already in that directory:

```bash
cd code_publish/eval_sccm_model
```

## 1. Download the Raw Data

Download the images and JSON annotations for the two required categories from [craigwu/vstar_bench](https://huggingface.co/datasets/craigwu/vstar_bench):

```bash
hf download craigwu/vstar_bench \
  --repo-type dataset \
  --local-dir data/raw/vstar_bench
```

This creates `data/raw/vstar_bench/direct_attributes/` and `data/raw/vstar_bench/relative_position/`, each containing `.jpg` images and matching `.json` annotations.

## 2. Generate the Evaluation Data

```bash
python data/recons_data.py
```

The script reads `data/raw/vstar_bench/` by default and generates the required evaluation file:

```text
data/processed/vstar_bench_format.parquet
```

The message `Read verification succeeded: ... samples` confirms that the Parquet file has been generated and loaded successfully. You can then run evaluation.

If the dataset is already downloaded, skip step 1 and specify its root directory, which must directly contain the two category subdirectories:

```bash
python data/recons_data.py --input-dir /path/to/downloaded/vstar_bench
```

The output location remains unchanged. Use `--output-file /path/to/output.parquet` to select a different output location, but the evaluators still expect the default path shown above. Repeated runs overwrite the existing output file.

## 3. Required Artifacts

After completing the default workflow, keep the following directories and file:

```text
data/
├── raw/vstar_bench/
│   ├── direct_attributes/       # Raw JPG/JSON files
│   └── relative_position/       # Raw JPG/JSON files
└── processed/
    └── vstar_bench_format.parquet
```

The evaluators read the Parquet file, but images are still loaded from the raw data directory. Do not keep only the Parquet file or delete the original images. When moving the data, preserve the relative layout of `raw/` and `processed/`.

When `--input-dir` points to an external directory, images are not copied into `data/raw/`. That input directory must also remain available.
