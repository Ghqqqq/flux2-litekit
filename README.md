# `Flux2 LiteKit`

Lightweight training and inference toolkit based on FLUX.2.

`Flux2 LiteKit` provides lightweight LoRA workflows for:

- text-to-image (`t2i`)
- image-to-image (`i2i`)

The project is intentionally domain-neutral. It expects generic image datasets and JSONL metadata files, and it exposes LoRA bootstrapping from a previous adapter as a first-class feature.

## Features

- Unified training CLI for `t2i` and `i2i`
- Unified inference CLI for `t2i` and `i2i`
- LoRA initialization from a previous LoRA adapter
- Optional local transformer override
- Stock `Flux2KleinPipeline` inference semantics for `i2i`

## Source Layout

```text
flux2-litekit/
├── flux2_litekit/
│   ├── __init__.py
│   ├── common.py
│   ├── config.py
│   ├── datasets.py
│   ├── infer.py
│   ├── lora.py
│   └── train.py
├── configs/
├── scripts/
└── examples/
```

## Installation

From the repository root:

```bash
pip install -e .
```

## Dataset Schemas

### Text-to-image

Directory layout:

```text
train_dir/
├── 0000000.png
├── 0000001.png
└── ...
```

Metadata format:

```json
{"file_name": "0000000.png", "text": "A clean product render on a studio background."}
```

### Image-to-image

Directory layout:

```text
train_root/
├── cond/
│   ├── 0000000.png
│   └── ...
├── target/
│   ├── 0000000.png
│   └── ...
└── metadata.jsonl
```

Metadata format:

```json
{"cond_file": "cond/0000000.png", "target_file": "target/0000000.png", "text": "Generate an output image that matches the reference content and style constraints."}
```

## Training

Run text-to-image training:

```bash
python -m flux2_litekit.train \
  --task t2i \
  --config configs/train_t2i.example.yaml
```

Or use the helper script:

```bash
bash scripts/run_train_t2i.sh
```

Run image-to-image training:

```bash
python -m flux2_litekit.train \
  --task i2i \
  --config configs/train_i2i.example.yaml
```

Or use the helper script:

```bash
bash scripts/run_train_i2i.sh
```

### Previous LoRA Bootstrap

To initialize a fresh LoRA from a previous adapter, set:

```yaml
bootstrap:
  init_lora_path: /path/to/previous/lora_weights
```

Leave the field empty to start from a fresh LoRA initialization.

Common examples:

```yaml
# Continue training a new t2i LoRA from a previous t2i adapter.
bootstrap:
  init_lora_path: ./outputs/previous_t2i_run/final/lora_weights
```

```yaml
# Start i2i training from an existing t2i adapter.
bootstrap:
  init_lora_path: ./outputs/t2i_run/final/lora_weights
```

You can point `init_lora_path` to either:

- `final/lora_weights/`
- `checkpoint-<step>/lora_weights/`

Dedicated bootstrap config examples are included:

```bash
python -m flux2_litekit.train \
  --task t2i \
  --config configs/train_t2i_bootstrap.example.yaml

python -m flux2_litekit.train \
  --task i2i \
  --config configs/train_i2i_bootstrap.example.yaml
```

## Inference

Run text-to-image inference:

```bash
python -m flux2_litekit.infer \
  --task t2i \
  --config configs/infer_t2i.example.yaml
```

Or use the helper script:

```bash
bash scripts/run_infer_t2i.sh
```

Run image-to-image inference:

```bash
python -m flux2_litekit.infer \
  --task i2i \
  --config configs/infer_i2i.example.yaml
```

Or use the helper script:

```bash
bash scripts/run_infer_i2i.sh
```

### Script Overrides

The helper scripts support a few common environment overrides:

- `CONFIG`: replace the default YAML path
- `CUDA_VISIBLE_DEVICES`: choose the visible GPU set
- `NUM_PROCESSES`: override `accelerate launch --num_processes`
- `MIXED_PRECISION`: override the training launch precision
- `RESUME_FROM`: pass an accelerate checkpoint directory to the training scripts
- `TRAIN_EXTRA_ARGS`: append additional arguments to the training CLI
- `INFER_EXTRA_ARGS`: append additional arguments to the inference CLI
- `PYTHON_BIN`: choose the Python interpreter used by the inference scripts

## Publish to GitHub

Typical flow:

```bash
cd opensource_flux2finetune
git init
git add .
git commit -m "Initial open-source release"
git branch -M main
git remote add origin git@github.com:<your-name>/flux2-litekit.git
git push -u origin main
```

If you prefer the local directory name to match the repository name before publishing:

```bash
mv opensource_flux2finetune flux2-litekit
cd flux2-litekit
```

## Development Checks

Local syntax checks:

```bash
python -m py_compile flux2_litekit/*.py
bash -n scripts/*.sh
```

## Notes

- `i2i` training in this package follows the stock `Flux2KleinPipeline` img2img contract.
- Adapter outputs are saved under `checkpoint-*/lora_weights/` and `final/lora_weights/`.
- Example metadata files are included under `examples/` for reference only.
- Recommended GitHub repository name: `flux2-litekit`
