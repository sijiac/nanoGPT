# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

nanoGPT is a minimalist implementation for training/finetuning medium-sized GPT models. The codebase prioritizes simplicity and readability over education - `train.py` (~300 lines) and `model.py` (~300 lines) contain the core functionality.

## Architecture

### Core Files

- **`model.py`**: Complete GPT model definition (~300 lines)
  - `GPTConfig`: Dataclass for model hyperparameters (n_layer, n_head, n_embd, block_size, vocab_size, dropout, bias)
  - `GPT`: Main model class with transformer blocks
  - `CausalSelfAttention`: Self-attention with Flash Attention support (PyTorch >= 2.0)
  - `Block`: Transformer block (LayerNorm → Attention → LayerNorm → MLP)
  - `GPT.from_pretrained()`: Load OpenAI GPT-2 weights ('gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl')
  - `GPT.generate()`: Autoregressive generation with temperature and top-k sampling

- **`train.py`**: Training loop with DDP support (~300 lines)
  - Supports three initialization modes: 'scratch', 'resume', or 'gpt2*'
  - Uses `get_batch()` with np.memmap for data loading (train.bin/val.bin format)
  - Cosine learning rate decay with linear warmup (`get_lr()`)
  - Saves checkpoints to `--out_dir` with model state, optimizer state, and config
  - DDP detection via RANK environment variable

- **`sample.py`**: Inference script
  - Load from checkpoint (`--out_dir`) or pretrained GPT-2 (`--init_from=gpt2*`)
  - Supports file-based prompts: `--start=FILE:prompt.txt`

- **`configurator.py`**: Simple config override system
  - Config files (in `config/`) set default hyperparameters via Python exec
  - Command-line args override config: `python train.py config/train_gpt2.py --batch_size=32`

### Data Format

Data should be prepared as binary files:
- `train.bin` and `val.bin`: Raw uint16 arrays of token IDs
- `meta.pkl` (optional): Dict with 'vocab_size' and optionally 'stoi'/'itos' for encoding/decoding
- Prepare scripts in `data/*/prepare.py` handle dataset-specific tokenization

## Common Commands

### Training

**Single GPU:**
```bash
python train.py --batch_size=32 --compile=False
```

**Character-level Shakespeare (quick start):**
```bash
python data/shakespeare_char/prepare.py
python train.py config/train_shakespeare_char.py
```

**Reproduce GPT-2 (124M) on 8x A100:**
```bash
python data/openwebtext/prepare.py
torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py
```

**Multi-node DDP (e.g. 2 nodes, 8 GPUs each):**
```bash
# Master node (123.456.123.456):
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py

# Worker node:
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
```

**Without Infiniband:** Prepend `NCCL_IB_DISABLE=1` to training commands

**CPU or Apple Silicon:**
```bash
# CPU
python train.py config/train_shakespeare_char.py --device=cpu --compile=False

# Apple M1/M2 (Metal Performance Shaders)
python train.py config/train_shakespeare_char.py --device=mps
```

### Finetuning

```bash
python data/shakespeare/prepare.py
python train.py config/finetune_shakespeare.py
```

Finetune configs set `init_from='gpt2'` (or 'gpt2-medium', 'gpt2-large', 'gpt2-xl') and use smaller learning rates.

### Sampling

**From trained checkpoint:**
```bash
python sample.py --out_dir=out-shakespeare-char
```

**From pretrained GPT-2:**
```bash
python sample.py --init_from=gpt2-xl --start="What is the answer to life, the universe, and everything?" --num_samples=5 --max_new_tokens=100
```

**From file prompt:**
```bash
python sample.py --start=FILE:prompt.txt
```

### Evaluation

Evaluate pretrained OpenAI models on OpenWebText:
```bash
python train.py config/eval_gpt2.py
python train.py config/eval_gpt2_medium.py
python train.py config/eval_gpt2_large.py
python train.py config/eval_gpt2_xl.py
```

### Benchmarking

```bash
python bench.py
```

`bench.py` isolates the training loop core for profiling without complexities.

## Key Training Parameters

Critical hyperparameters in `train.py` (can be overridden via config files or CLI):

- **Model**: `n_layer`, `n_head`, `n_embd`, `block_size`, `dropout`, `bias`
- **Data**: `dataset`, `batch_size`, `gradient_accumulation_steps`
- **Optimizer**: `learning_rate`, `weight_decay`, `beta1`, `beta2`, `grad_clip`
- **LR Schedule**: `warmup_iters`, `lr_decay_iters`, `min_lr`, `decay_lr`
- **Training**: `max_iters`, `eval_interval`, `eval_iters`, `compile`
- **System**: `device` ('cuda', 'cpu', 'mps'), `dtype` ('float32', 'bfloat16', 'float16'), `backend` ('nccl', 'gloo')

## Configuration System

The "Poor Man's Configurator" (`configurator.py`):
1. Config files are Python scripts that set globals (e.g., `config/train_gpt2.py`)
2. Run: `python train.py config/train_gpt2.py --batch_size=32`
3. Config file executes first, then CLI args override specific values

## Important Implementation Details

- **Data Loading**: Uses `np.memmap` recreated each batch to avoid memory leaks
- **DDP**: Detected via `RANK` env var; adjusts gradient_accumulation_steps automatically
- **Checkpointing**: Saves to `{out_dir}/ckpt.pt` with model, optimizer, config, iter_num, best_val_loss
- **Flash Attention**: Automatically enabled if PyTorch >= 2.0 (via `torch.nn.functional.scaled_dot_product_attention`)
- **torch.compile**: Enabled by default (`--compile=True`) for ~2x speedup; disable on Windows or if errors occur
- **Weight Tying**: Token embeddings and LM head weights are tied (`self.transformer.wte.weight = self.lm_head.weight`)
- **Gradient Accumulation**: Simulates larger batch sizes; actual tokens/iter = `gradient_accumulation_steps * ddp_world_size * batch_size * block_size`

## Troubleshooting

- **PyTorch 2.0 compile errors**: Add `--compile=False` (slower but compatible)
- **OOM on GPU**: Reduce `batch_size`, `block_size`, or model size
- **Slow multi-node training**: Check interconnect (use `iperf3`); without Infiniband, prepend `NCCL_IB_DISABLE=1`
- **Checkpoint loading issues**: Check for `_orig_mod.` prefix in state_dict (handled automatically in train.py/sample.py)

## Directory Structure

- `config/`: Training configuration files (Python scripts)
- `data/`: Dataset-specific prepare.py scripts and binary data (train.bin, val.bin, meta.pkl)
- `out*/`: Training outputs (checkpoints, logs) - directory specified by `--out_dir`
