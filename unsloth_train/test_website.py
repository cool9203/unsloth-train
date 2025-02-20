# coding: utf-8

import argparse
import base64
import io
import time
import traceback
from pathlib import Path

import gradio as gr
import httpx
import pypandoc
import torch
from peft import PeftConfig, PeftModel
from PIL import Image
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    ProcessorMixin,
)

from unsloth_train import utils

_default_prompt = "latex table ocr"
_default_system_prompt = "You should follow the instructions carefully and explain your answers in detail."

__model: dict[str, PreTrainedModel | ProcessorMixin | PreTrainedTokenizer | str] = {
    "model": None,
    "tokenizer": None,
    "name": None,
}


def arg_parser() -> argparse.Namespace:
    """取得執行程式時傳遞的參數

    tutorial: https://docs.python.org/zh-tw/3/howto/argparse.html#
    reference: https://docs.python.org/zh-tw/3/library/argparse.html#nargs

    Returns:
        argparse.Namespace: 使用args.name取得傳遞的參數
    """

    parser = argparse.ArgumentParser(description="Run test website to test unsloth training with llm or vlm")
    parser.add_argument("-m", "--model_name", type=str, default=None, help="Run model name or path")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Web server host")
    parser.add_argument("--port", type=int, default=7860, help="Web server port")
    parser.add_argument("--max_tokens", type=int, default=4096, help="Run model generate max new tokens")
    parser.add_argument("--device_map", type=str, default="cuda:0", help="Run model device map")
    parser.add_argument("--dev", dest="dev_mode", action="store_true", help="Dev mode")
    parser.add_argument("--example_folder", type=str, default="example", help="Example folder")

    args = parser.parse_args()

    return args


def load_model(
    model_name: str,
    load_in_4bit: bool = True,
    device_map: str = "cuda:0",
    token: int = 8192,
    revision: str = None,
    trust_remote_code: bool = False,
):
    try:
        model = AutoModelForVision2Seq.from_pretrained(
            pretrained_model_name_or_path=model_name,
            device_map=device_map,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=load_in_4bit,
                bnb_4bit_compute_dtype=torch.bfloat16,
            ),
            attn_implementation="flash_attention_2",
        )
    except ValueError:
        model = AutoModelForVision2Seq.from_pretrained(
            pretrained_model_name_or_path=model_name,
            device_map=device_map,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=load_in_4bit,
                bnb_4bit_compute_dtype=torch.bfloat16,
            ),
        )
    tokenizer = AutoProcessor.from_pretrained(
        pretrained_model_name_or_path=model_name,
        device_map=device_map,
    )

    try:
        PeftConfig.from_pretrained(
            model_name,
            token=token,
            revision=revision,
            trust_remote_code=trust_remote_code,
        )
        model = PeftModel.from_pretrained(
            model,
            model_name,
            token=token,
            revision=revision,
            is_trainable=True,
            trust_remote_code=trust_remote_code,
        )
    except Exception:
        pass

    # For inference mode
    model.gradient_checkpointing = False
    model.training = False
    for name, module in model.named_modules():
        if hasattr(module, "gradient_checkpointing"):
            module.gradient_checkpointing = False
        if hasattr(module, "training"):
            module.training = False

    return (model, tokenizer)


def generate(
    image,
    prompt: str,
    system_prompt: str = _default_system_prompt,
    device_map: str = "auto",
    max_new_tokens: int = 1024,
    **kwds,
) -> dict[str, str | int]:
    model = __model.get("model")
    tokenizer = __model.get("tokenizer")
    messages = list()

    if system_prompt:
        messages.append(
            {
                "role": "user",
                "content": [{"type": "text", "text": system_prompt}],
            }
        )
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    )
    input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    inputs = tokenizer(
        image,
        input_text,
        add_special_tokens=True,
        return_tensors="pt",
    ).to(device_map)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        **kwds,
    )

    # Reference: https://github.com/huggingface/transformers/issues/17117#issuecomment-1124497554
    return {
        "content": tokenizer.batch_decode(outputs[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True)[0],
        "usage": {
            "prompt_tokens": inputs["input_ids"].shape[1],
            "completion_tokens": len(outputs[:, inputs["input_ids"].shape[1] :][0]),
            "total_tokens": inputs["input_ids"].shape[1] + len(outputs[:, inputs["input_ids"].shape[1] :][0]),
        },
    }


def inference_table(
    image: Image.Image,
    prompt: str,
    detect_table: bool,
    crop_table_padding: int,
    max_tokens: int = 4096,
    model_name: str = None,
    system_prompt: str = _default_system_prompt,
    device_map: str = "auto",
    repair_latex: bool = False,
    full_border: bool = False,
    unsqueeze: bool = False,
):
    origin_responses = list()
    crop_images = list()
    used_time = 0
    completion_tokens = 0

    if model_name and model_name != __model.get("name", None):
        (__model["model"], __model["tokenizer"]) = load_model(
            model_name=model_name,
            device_map=device_map,
        )
        __model["name"] = model_name

    if detect_table:
        with io.BytesIO() as file_io:
            image.save(file_io, format="png")
            file_io.seek(0)
            resp = httpx.post(
                "http://10.70.0.232:9999/upload",
                files={"file": ("image.png", file_io)},
                data={
                    "action": "crop",
                    "padding": crop_table_padding,
                },
            )

        for crop_image_base64 in resp.json():
            crop_image_data = base64.b64decode(crop_image_base64)
            crop_images.append(Image.open(io.BytesIO(crop_image_data)))
    else:
        crop_images.append(image)

    try:
        for crop_image in crop_images:
            start_time = time.time()
            generate_response = generate(
                prompt=prompt,
                system_prompt=system_prompt,
                image=crop_image,
                device_map=device_map,
                max_new_tokens=max_tokens,
                use_cache=True,
                top_p=1.0,
                top_k=None,
                do_sample=False,
                temperature=None,
            )
            end_time = time.time()
            used_time += end_time - start_time
            completion_tokens += generate_response["usage"]["completion_tokens"]

            if repair_latex:
                origin_responses.append(
                    utils.convert_pandas_to_latex(
                        df=utils.convert_latex_table_to_pandas(
                            latex_table_str=generate_response["content"],
                            headers=True,
                            unsqueeze=unsqueeze,
                        ),
                        full_border=full_border,
                    )
                )
            else:
                origin_responses.append(generate_response["content"])
        try:
            html_response = pypandoc.convert_text("".join(origin_responses), "html", format="latex")
        except Exception:
            try:
                html_response = pypandoc.convert_text("".join(origin_responses), "html", format="markdown")
            except Exception:
                html_response = pypandoc.convert_text("".join(origin_responses), "html", format="html")
    except Exception as e:
        html_response = "推論輸出無法解析"
        traceback.print_exception(e)

    return (
        "\n\n".join(origin_responses),
        html_response,
        crop_images,
        completion_tokens / (used_time if used_time > 0 else 1e-6),
    )


def test_website(
    host: str = "127.0.0.1",
    port: int = 7860,
    model_name: str = None,
    max_tokens: int = 4096,
    device_map: str = "cuda:0",
    dev_mode: bool = False,
    example_folder: str = "examples",
    **kwds,
):
    if model_name and __model.get("name") is None:
        (__model["model"], __model["tokenizer"]) = load_model(
            model_name=model_name,
            device_map=device_map,
        )
        __model["name"] = model_name

    # Gradio 接口定義
    with gr.Blocks(
        title="VLM 生成表格測試網站",
        css="#component-6 { max-height: 85vh; }",
    ) as demo:
        gr.Markdown("## VLM 生成表格測試網站")

        with gr.Row():
            with gr.Column():
                image_input = gr.Image(
                    label="上傳圖片",
                    type="pil",
                    height="85vh",
                )

            with gr.Column():
                html_output = gr.HTML(label="生成的表格輸出")

        submit_button = gr.Button("生成表格")

        with gr.Row():
            with gr.Column():
                crop_table_results = gr.Gallery(label="偵測表格結果", format="png")

            with gr.Column():
                _model_name = gr.Textbox(label="模型名稱或路徑", value=__model.get("name", None), visible=not model_name)
                system_prompt_input = gr.Textbox(label="輸入系統文字提示", lines=2, value=_default_system_prompt)
                prompt_input = gr.Textbox(label="輸入文字提示", lines=2, value=_default_prompt)
                _max_tokens = gr.Slider(label="Max tokens", value=max_tokens, minimum=1, maximum=8192, step=1)
                detect_table = gr.Checkbox(label="是否自動偵測表格", value=True)
                crop_table_padding = gr.Slider(label="偵測表格裁切框 padding", value=-60, minimum=-300, maximum=300, step=1)
                repair_latex = gr.Checkbox(value=True, label="修復 latex", visible=dev_mode)
                full_border = gr.Checkbox(label="修復 latex 表格全框線", visible=dev_mode)
                unsqueeze = gr.Checkbox(label="修復 latex 並解開多行/列合併", visible=dev_mode)
                time_usage = gr.Textbox(label="每秒幾個 token")

        text_output = gr.Textbox(label="生成的文字輸出", visible=dev_mode)

        # Constant augments
        _device_map = gr.Textbox(value=device_map, visible=False)

        # Examples
        if Path(example_folder).exists():
            example_files = sorted(
                [
                    (str(path.resolve()), path.name)
                    for path in Path(example_folder).iterdir()
                    if path.suffix.lower() in [".jpg", ".jpeg", ".png"]
                ],
                key=lambda e: e[1],
            )
            examples = gr.Examples(
                examples=[
                    [
                        Image.open(path),
                        _default_prompt,
                        True,
                        -60,
                        4096,
                        _model_name,
                        _default_system_prompt,
                        _device_map,
                        True,
                        False,
                        False,
                    ]
                    for path, name in example_files
                ],
                example_labels=[name for path, name in example_files],
                inputs=[
                    image_input,
                    prompt_input,
                    detect_table,
                    crop_table_padding,
                    _max_tokens,
                    _model_name,
                    system_prompt_input,
                    _device_map,
                    repair_latex,
                    full_border,
                    unsqueeze,
                ],
            )

        submit_button.click(
            inference_table,
            inputs=[
                image_input,
                prompt_input,
                detect_table,
                crop_table_padding,
                _max_tokens,
                _model_name,
                system_prompt_input,
                _device_map,
                repair_latex,
                full_border,
                unsqueeze,
            ],
            outputs=[
                text_output,
                html_output,
                crop_table_results,
                time_usage,
            ],
        )
        demo.launch(
            server_name=host,
            server_port=port,
            share=False,  # Reference: https://github.com/gradio-app/gradio/issues/7978#issuecomment-2567283591
        )


if __name__ == "__main__":
    args = arg_parser()
    args_dict = vars(args)
    test_website(**args_dict)
