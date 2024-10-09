# coding: utf-8

import argparse
from pathlib import Path
from unittest.mock import patch

import unsloth.save
from _patch import create_ollama_modelfile


def arg_parser() -> argparse.Namespace:
    """取得執行程式時傳遞的參數

    tutorial: https://docs.python.org/zh-tw/3/howto/argparse.html#
    reference: https://docs.python.org/zh-tw/3/library/argparse.html#nargs

    Returns:
        argparse.Namespace: 使用args.name取得傳遞的參數
    """

    parser = argparse.ArgumentParser(description="Dry run save to gguf model.")

    parser.add_argument("-m", "--model_name", help="base llm model")

    args = parser.parse_args()

    return args


@patch.object(unsloth.save, "create_ollama_modelfile", create_ollama_modelfile)
def save_to_gguf(
    model_name: str,
    save_path: str,
    save_model_name: str,
    quantization_method: str,
    chat_template_name: str = None,
    **kwds,
) -> None:
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
    )

    _model_name = model_name.lower()
    if "llama-3.1" in _model_name or "llama3.1" in _model_name:
        chat_template_name = "llama-3.1"
    elif "llama-3.2" in _model_name or "llama3.2" in _model_name:
        chat_template_name = "llama-3.2"
    elif "qwen-2.5" in _model_name or "qwen2.5" in _model_name:
        chat_template_name = "qwen-2.5"
    elif not chat_template_name:
        raise ValueError(f"Model of '{model_name}' not support auto select chat_template")

    tokenizer = get_chat_template(
        tokenizer,
        chat_template=chat_template_name,
    )

    save_model_path = Path(save_path, save_model_name)
    save_model_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained_gguf(
        f"{str(save_model_path)}",
        tokenizer,
        quantization_method=quantization_method,
        maximum_memory_usage=0.75,
    )


if __name__ == "__main__":
    # args = vars(arg_parser())
    # save_to_gguf(**args)

    save_to_gguf(
        model_name="./outputs/7B-checkpoint/checkpoint-1015",
        save_path="models",
        save_model_name="test-qwen-2.5-7B-format_3-epoch_5-lr_7e06-context_length_1024",
        quantization_method=["f32", "q4_k_m"],
        chat_template_name="qwen-2.5",
    )
