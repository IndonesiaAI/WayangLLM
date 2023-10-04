## Prerequisites

Install all the dependencies in the `requirements.txt`:

```
$ pip install -U -r requirements.txt
```

On multi-GPU, run:
```
$ accelerate config
```

## Training

There were two main steps to the DPO training process:
1. Supervised fine-tuning of the base llama-v2-7b model to create llama-v2-7b-se:
    - `accelerate launch sft.py --output_dir="sft"`
2. Run the DPO trainer using the model saved by the previous step:
    - `accelerate launch dpo.py --model_name_or_path="sft/final_checkpoint" --output_dir="dpo"`

On single GPU, run:
1. Supervised fine-tuning of the base llama-v2-7b model to create llama-v2-7b-se:
    - `python sft.py --output_dir="sft"`
2. Run the DPO trainer using the model saved by the previous step:
    - `python dpo.py --model_name_or_path="sft/final_checkpoint" --output_dir="dpo"`

Merge the adaptors into the base model:

```sh
python merge_peft_adapter.py --base_model_name="meta-llama/Llama-2-7b-hf" --adapter_model_name="dpo/final_checkpoint/" --output_name="stack-llama-2"
```
