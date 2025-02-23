# coding: utf-8

import os
from pathlib import Path
from typing import Any, Callable, Dict, List
from unittest.mock import patch

import unsloth.tokenizer_utils

from unsloth_train._patch import fix_chat_template
from unsloth_train.utils import _get_function_used_params

os.environ["UNSLOTH_RETURN_LOGITS"] = "1"  # Reference: https://github.com/unslothai/unsloth/issues/1417#issuecomment-2538266052


@patch.object(unsloth.tokenizer_utils, "fix_chat_template", fix_chat_template)
def train_model(
    output_model_path: str,
    model_name: str,
    dataset_path: str,
    make_dataset_fn: Callable,
    make_dataset_parameters: Dict[str, Any] = {},
    quantization_method: str = ["q4_k_m"],
    seed: int = 3407,
    max_seq_length: int = 2048,
    chat_template_name: str = None,
    dtype: str = None,  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit: bool = True,  # Use 4bit quantization to reduce memory usage. Can be False.
    instruction_part: str = None,
    response_part: str = None,
    num_train_epochs: int = 1,
    learning_rate: float = 2e-4,
    target_modules: List[str] = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    checkpoint_path: str = "outputs",
):
    import torch
    from transformers import DataCollatorForSeq2Seq, TrainingArguments
    from trl import SFTTrainer
    from unsloth import FastLanguageModel, is_bfloat16_supported
    from unsloth.chat_templates import get_chat_template, train_on_responses_only

    # torch.backends.cuda.enable_cudnn_sdp(False)  # Fix newest nvidia gpu, like A6000

    model, tokenizer = FastLanguageModel.from_pretrained(
        # model_name="unsloth/Llama-3.2-1B-Instruct",  # or choose "unsloth/Llama-3.2-1B-Instruct"
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
    )

    try:
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
            target_modules=target_modules,
            lora_alpha=16,
            lora_dropout=0,  # Supports any, but = 0 is optimized
            bias="none",  # Supports any, but = "none" is optimized
            # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
            use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
            random_state=seed,
            use_rslora=False,  # We support rank stabilized LoRA
            loftq_config=None,  # And LoftQ
        )
    except RuntimeError as e:
        if "added LoRA adapter".lower() not in str(e).lower():
            raise e

    _model_name = model_name.lower()
    if "llama-3.1" in _model_name or "llama3.1" in _model_name or "llama-3.2" in _model_name or "llama3.2" in _model_name:
        chat_template_name = "llama-3.1"
    elif "qwen-2.5" in _model_name or "qwen2.5" in _model_name:
        chat_template_name = "qwen-2.5"
    elif not chat_template_name:
        raise ValueError(f"Model of '{model_name}' not support auto select chat_template")

    tokenizer = get_chat_template(
        tokenizer,
        chat_template=chat_template_name,
    )

    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
        return {
            "text": texts,
        }

    make_dataset_parameters = _get_function_used_params(
        make_dataset_fn,
        seed=seed,
        **make_dataset_parameters,
    )

    dataset = make_dataset_fn(
        dataset_path=dataset_path,
        **make_dataset_parameters,
    )
    dataset = dataset.map(
        formatting_prompts_func,
        batched=True,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
        dataset_num_proc=2,
        packing=False,  # Can make training 5x faster for short sequences.
        args=TrainingArguments(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            num_train_epochs=num_train_epochs,  # Set this for 1 full training run.
            # max_steps=60,
            learning_rate=learning_rate,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=seed,
            output_dir=checkpoint_path,
        ),
    )

    if chat_template_name == "llama-3.1":
        trainer = train_on_responses_only(
            trainer,
            instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
            response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
        )
    elif chat_template_name == "qwen-2.5":
        trainer = train_on_responses_only(
            trainer,
            instruction_part="<|im_start|>user\n",
            response_part="<|im_start|>assistant\n",
        )
    elif not instruction_part and not response_part:
        raise ValueError(f"Model of '{model_name}' not support auto select instruction_part and response_part")

    trainer_stats = trainer.train()

    output_model_path = Path(output_model_path)
    output_model_path.mkdir(parents=True, exist_ok=True)

    try:
        model.save_pretrained_gguf(
            f"{str(output_model_path)}",
            tokenizer,
            quantization_method=quantization_method,
            maximum_memory_usage=0.75,
        )
    except Exception:
        pass
    # Save - Transformers on local
    model.save_pretrained(f"{str(output_model_path)}/lora_model")
    tokenizer.save_pretrained(f"{str(output_model_path)}/lora_model")


if __name__ == "__main__":
    learning_rate = 7e-6
    epoch = 5
    max_seq_length = 2048
    train_model(
        model_name="shenzhi-wang/Llama3.1-8B-Chinese-Chat",
        dataset_path="/mnt/d/dataset/finance/金科QA整理-20240926.xlsx",
        max_seq_length=max_seq_length,
        save_path="/mnt/d/models",
        save_model_name=f"Llama3.1-8B-Chinese-Chat-finance-qa-v3-context_length_{max_seq_length}",
        quantization_method=["q4_k_m"],
        num_train_epochs=epoch,
        learning_rate=learning_rate,
    )
