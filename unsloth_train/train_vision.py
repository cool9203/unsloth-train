# coding: utf-8

import os
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
)
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
    seed: int = 3407,
    max_seq_length: int = 2048,
    load_in_4bit: bool = True,  # Use 4bit quantization to reduce memory usage. Can be False.
    num_train_epochs: int = 1,
    learning_rate: float = 2e-4,
    finetune_vision_layers: bool = False,
    finetune_language_layers: bool = True,
    finetune_attention_modules: bool = True,
    finetune_mlp_modules: bool = True,
    target_modules: str = None,
    checkpoint_path: str = "outputs",
):
    import torch
    from _patch import UnslothVisionModelTextDataCollator
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastVisionModel, is_bf16_supported
    from unsloth_zoo.vision_utils import UnslothVisionDataCollator

    torch.backends.cuda.enable_cudnn_sdp(False)  # Fix newest nvidia gpu, like A6000

    model, tokenizer = FastVisionModel.from_pretrained(
        model_name,
        load_in_4bit=load_in_4bit,  # Use 4bit to reduce memory use. False for 16bit LoRA.
        use_gradient_checkpointing="unsloth",  # True or "unsloth" for long context
    )

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=finetune_vision_layers,  # False if not finetuning vision layers
        finetune_language_layers=finetune_language_layers,  # False if not finetuning language layers
        finetune_attention_modules=finetune_attention_modules,  # False if not finetuning attention layers
        finetune_mlp_modules=finetune_mlp_modules,  # False if not finetuning MLP layers
        r=16,  # The larger, the higher the accuracy, but might overfit
        lora_alpha=16,  # Recommended alpha == r at least
        lora_dropout=0,
        bias="none",
        random_state=seed,
        use_rslora=False,  # We support rank stabilized LoRA
        loftq_config=None,  # And LoftQ
        target_modules=target_modules,  # Optional now! Can specify a list if needed
    )

    def convert_to_conversation(sample):
        messages = []
        added_image = False
        for message in sample["conversations"]:
            if message["role"] in ["assistant"]:
                messages.append({"role": message["role"], "content": [{"type": "text", "text": message["content"]}]})
            elif message["role"] in ["user"] and not added_image:
                added_image = True
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": message["content"]},
                        ],
                    }
                )
            elif message["role"] in ["user", "system"]:
                messages.append({"role": "user", "content": [{"type": "text", "text": message["content"]}]})

        return {"messages": messages}

    make_dataset_parameters = _get_function_used_params(
        make_dataset_fn,
        seed=seed,
        **make_dataset_parameters,
    )

    dataset = make_dataset_fn(
        dataset_path=dataset_path,
        **make_dataset_parameters,
    )
    converted_dataset = [convert_to_conversation(sample) for sample in dataset]

    try:
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            data_collator=UnslothVisionDataCollator(model, tokenizer),  # Must use!
            train_dataset=converted_dataset,
            args=SFTConfig(
                per_device_train_batch_size=1,
                gradient_accumulation_steps=4,
                warmup_steps=5,
                num_train_epochs=num_train_epochs,  # Set this for 1 full training run.
                learning_rate=learning_rate,
                fp16=not is_bf16_supported(),
                bf16=is_bf16_supported(),
                logging_steps=1,
                optim="adamw_8bit",
                weight_decay=0.01,
                lr_scheduler_type="linear",
                seed=seed,
                output_dir=checkpoint_path,
                report_to="none",  # For Weights and Biases
                # You MUST put the below items for vision finetuning:
                remove_unused_columns=False,
                dataset_text_field="",
                dataset_kwargs={"skip_prepare_dataset": True},
                dataset_num_proc=2,
                max_seq_length=max_seq_length,
            ),
        )
        trainer_stats = trainer.train()
    except ValueError:  # Fix data must be have a image, will backward to only text tokenizer
        print("Backward to 'TextTokenizer'")
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            data_collator=UnslothVisionModelTextDataCollator(model, tokenizer),  # Must use!
            train_dataset=converted_dataset,
            args=SFTConfig(
                per_device_train_batch_size=1,
                gradient_accumulation_steps=4,
                warmup_steps=5,
                num_train_epochs=num_train_epochs,  # Set this for 1 full training run.
                learning_rate=learning_rate,
                fp16=not is_bf16_supported(),
                bf16=is_bf16_supported(),
                logging_steps=1,
                optim="adamw_8bit",
                weight_decay=0.01,
                lr_scheduler_type="linear",
                seed=seed,
                output_dir=checkpoint_path,
                report_to="none",  # For Weights and Biases
                # You MUST put the below items for vision finetuning:
                remove_unused_columns=False,
                dataset_text_field="",
                dataset_kwargs={"skip_prepare_dataset": True},
                dataset_num_proc=2,
                max_seq_length=max_seq_length,
            ),
        )

        trainer_stats = trainer.train()

    output_model_path = Path(output_model_path)
    output_model_path.mkdir(parents=True, exist_ok=True)
    # Save - Transformers on local
    model.save_pretrained(f"{str(output_model_path)}/lora_model")
    tokenizer.save_pretrained(f"{str(output_model_path)}/lora_model")
    model.save_pretrained_merged(f"{str(output_model_path)}/transformers", tokenizer)


if __name__ == "__main__":
    learning_rate = 7e-6
    epoch = 3
    max_seq_length = 4096
    train_model(
        model_name="unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit",
        dataset_path="/mnt/d/dataset/finance/金科QA整理-20240926.xlsx",
        max_seq_length=max_seq_length,
        output_model_path=f"/mnt/d/models/Llama3.2-11B-Vision-Instruct-context_length_{max_seq_length}",
        num_train_epochs=epoch,
        learning_rate=learning_rate,
    )
