# WayangLLM

```bash
pip install torch
```

```bash
git clone https://github.com/IndonesiaAI/WayangLLM/ -b zephyr
```

```bash
cd WayangLLM
```

```bash
python -m pip install .
```

```bash
MAX_JOBS=4 pip install flash-attn --no-build-isolation
```

```bash
huggingface-cli login --token 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
```

```bash
sudo apt-get install git-lfs
```

```bash
# Step 1 - SFT
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/multi_gpu.yaml --num_processes=1 scripts/run_sft.py recipes/zephyr-7b-beta/sft/config_lora.yaml

# Step 2 - DPO
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/multi_gpu.yaml --num_processes=1 scripts/run_dpo.py recipes/zephyr-7b-beta/dpo/config_lora.yaml
```