# coding: utf-8

from __future__ import print_function

import torch
from unsloth.tokenizer_utils import logger
from unsloth_zoo.vision_utils import (
    _get_dtype,
    get_padding_tokens_ids,
    process_vision_info,
)


def _fix_chat_template(chat_template):
    endfor = "{% endfor %}"
    where = chat_template.find(endfor)
    if where == -1:
        return chat_template

    after_endfor = chat_template[where + len(endfor) :]

    if (
        "{% if" not in after_endfor
        and "{% set " not in after_endfor
        and after_endfor.startswith("{{")
        and after_endfor.endswith("}}")
        and after_endfor.count("{{") == 1
        and after_endfor.count("}}") == 1
    ):
        after_endfor = "{% if add_generation_prompt %}" + after_endfor + "{% endif %}"

        chat_template = chat_template[: where + len(endfor)] + after_endfor
    pass
    return chat_template


def fix_chat_template(tokenizer):
    chat_template = getattr(tokenizer, "chat_template", None)
    if chat_template is None:
        return None

    ### 1. Check if add_generation_prompt works
    # Check for ShareGPT style first
    is_sharegpt = None
    try:
        messages = [
            {"role": "user", "content": "Who are you?"},
        ]
        tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
        is_sharegpt = False
    except Exception:
        try:
            messages = [
                {"from": "human", "value": "Who are you?"},
            ]
            tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
            is_sharegpt = True
        except Exception:
            is_sharegpt = None
        pass
    pass

    # Not ShareGPT or HF style - just return
    if is_sharegpt is None:
        return chat_template

    # Tokenize
    messages = [{"role": "user", "content": "Who are you?"} if not is_sharegpt else {"from": "human", "value": "Who are you?"}]
    no = tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
    yes = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    if no == yes:
        # SAME?! That's not good! We check for add_generation_prompt
        if "{% if add_generation_prompt %}" not in chat_template:
            # Try fixing it by adding it
            new_chat_template = _fix_chat_template(chat_template)
            if "{% if add_generation_prompt %}" not in new_chat_template:
                raise RuntimeError(
                    f"Unsloth: The tokenizer `{tokenizer.name_or_path}`\n"
                    "does not have a {% if add_generation_prompt %} for generation purposes.\n"
                    "Please file a bug report immediately - thanks!"
                )
            else:
                logger.warning_once(
                    "Unsloth: We successfully patched the tokenizer to add a {% if add_generation_prompt %} to the chat_template.\n"
                    "This is not a bug, but please notify the Unsloth maintainers - thanks!"
                )
                chat_template = new_chat_template
            pass
        else:
            raise RuntimeError(
                f"Unsloth: The tokenizer `{tokenizer.name_or_path}`\n"
                "has a {% if add_generation_prompt %} for generation purposes, but wasn't provided correctly.\n"
                "Please file a bug report immediately - thanks!"
            )
        pass
    pass
    return chat_template


class UnslothVisionDataCollator:
    __slots__ = "padding_token_ids", "dtype", "ignore_index", "processor"

    def __init__(self, model, processor, ignore_index=-100):
        self.padding_token_ids = get_padding_tokens_ids(processor)
        self.dtype = _get_dtype(
            model.config.torch_dtype if hasattr(model.config, "torch_dtype") else model.get_input_embeddings().weight.dtype
        )
        self.ignore_index = ignore_index
        self.processor = processor
        return

    def __call__(self, examples):
        # [TODO] Support non image inputs as well
        # The issue is batch = self.processor( forces tensors to be returned and not None.
        texts = []
        images = []
        for example in examples:
            messages = example["messages"]
            message = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            # Dataset with 2 columns messages / images
            if "images" in example:
                image = example["images"][0]
            else:
                image, video = process_vision_info(messages)
            texts.append(message)
            images.append(image)
        pass

        # Tokenize the texts and process the images
        batch = self.processor.tokenizer(
            text=texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        # FIXME: Support image processor
        # batch = self.processor.tokenizer(
        #     text=texts,
        #     images=images,
        #     padding=True,
        #     # [TODO] Truncating to max_seq_length does NOT work for VLMs
        #     truncation=True,
        #     return_tensors="pt",
        # )
        batch.pop("token_type_ids", None)

        # Pixtral accepts multiple images, so we have to cast it individually
        if "pixel_values" in batch:
            pixel_values = batch["pixel_values"]
            if type(pixel_values) is list:
                for j, pixel_value_j in enumerate(pixel_values):
                    if type(pixel_value_j) is list:
                        for k, pixel_value_k in enumerate(pixel_value_j):
                            pixel_value_j[k] = pixel_value_k.to(self.dtype)
                    else:
                        pixel_values[j] = pixel_value_j.to(self.dtype)
                pass
                batch["pixel_values"] = pixel_values
            else:
                batch["pixel_values"] = batch["pixel_values"].to(self.dtype)

        # Mask image tokens and pad tokens
        labels = batch["input_ids"].clone()
        labels[torch.isin(labels, self.padding_token_ids)] = self.ignore_index
        batch["labels"] = labels
        return batch
