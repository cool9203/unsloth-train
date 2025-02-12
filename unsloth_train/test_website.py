# coding: utf-8

import argparse
import base64
import io

import gradio as gr
import httpx
import pypandoc
from PIL import Image
from transformers import MllamaForConditionalGeneration, MllamaProcessor
from unsloth import FastVisionModel

from unsloth_train import utils

_default_prompt = "latex table ocr"
_default_system_prompt = "You should follow the instructions carefully and explain your answers in detail."

__model: dict[str, MllamaForConditionalGeneration | MllamaProcessor | str] = {
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
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Web server host")
    parser.add_argument("--port", type=int, default=7860, help="Web server port")
    parser.add_argument("--model_name", type=str, default=None, help="Run model name or path")
    parser.add_argument("--max_tokens", type=int, default=4096, help="Run model generate max new tokens")
    parser.add_argument("--device_map", type=str, default="cuda:0", help="Run model device map")

    args = parser.parse_args()

    return args


def load_model(
    model_name: str,
    load_in_4bit: bool = True,
    device_map: str = "cuda:0",
):
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=model_name,  # YOUR MODEL YOU USED FOR TRAINING
        load_in_4bit=load_in_4bit,  # Set to False for 16bit LoRA
        device_map=device_map,
    )
    FastVisionModel.for_inference(model)  # Enable for inference!
    return (model, tokenizer)


def generate(
    image,
    prompt: str,
    system_prompt: str = _default_system_prompt,
    device_map: str = "auto",
    max_new_tokens: int = 1024,
    **kwds,
) -> str:
    model = __model.get("model")
    tokenizer = __model.get("tokenizer")
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": system_prompt}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        },
    ]
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
    return tokenizer.batch_decode(outputs[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True)[0]


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
    crop_image = image
    file_io = io.BytesIO()
    image.save(file_io, format="png")
    file_io.seek(0)
    if model_name and model_name != __model.get("name", None):
        (__model["model"], __model["tokenizer"]) = load_model(
            model_name=model_name,
            device_map=device_map,
        )
        __model["name"] = model_name

    if detect_table:
        resp = httpx.post(
            "http://10.70.0.232:9999/upload",
            files={"file": file_io},
            data={
                "action": "crop",
                "padding": crop_table_padding,
            },
        )

        for _, crop_image_base64 in resp.json().items():
            crop_image_data = base64.b64decode(crop_image_base64)
            crop_image = Image.open(io.BytesIO(crop_image_data))
            break

    origin_response = generate(
        prompt=prompt,
        system_prompt=system_prompt,
        image=crop_image,
        device_map=device_map,
        max_new_tokens=max_tokens,
        use_cache=True,
        top_p=1.0,
        do_sample=False,
    )
    try:
        if repair_latex:
            origin_response = utils.convert_pandas_to_latex(
                df=utils.convert_latex_table_to_pandas(
                    latex_table_str=origin_response,
                    headers=True,
                    unsqueeze=unsqueeze,
                ),
                full_border=full_border,
            )
        html_response = pypandoc.convert_text(origin_response, "html", format="latex")
    except Exception as e:
        try:
            html_response = pypandoc.convert_text(origin_response, "html", format="markdown")
        except Exception as e:
            html_response = "輸出的內容不是正確的 latex or markdown"
    return origin_response, html_response, crop_image


def test_website(
    host: str = "127.0.0.1",
    port: int = 7860,
    model_name: str = None,
    max_tokens: int = 4096,
    device_map: str = "cuda:0",
    **kwds,
):
    if model_name and __model.get("name") is None:
        (__model["model"], __model["tokenizer"]) = load_model(
            model_name=model_name,
            device_map=device_map,
        )
        __model["name"] = model_name

    # Gradio 接口定義
    with gr.Blocks(title="VLM 生成表格測試網站") as demo:
        gr.Markdown("## VLM 生成表格測試網站")

        with gr.Row():
            with gr.Column():
                image_input = gr.Image(label="上傳圖片", type="pil")

            with gr.Column():
                _model_name = gr.Textbox(label="模型名稱或路徑", value=__model.get("name", None), visible=not model_name)
                system_prompt_input = gr.Textbox(label="輸入系統文字提示", lines=2, value=_default_system_prompt)
                prompt_input = gr.Textbox(label="輸入文字提示", lines=2, value=_default_prompt)
                _max_tokens = gr.Slider(label="Max tokens", value=max_tokens, minimum=1, maximum=8192, step=1)
                detect_table = gr.Checkbox(label="是否自動偵測表格", value=True)
                crop_table_padding = gr.Slider(label="偵測表格裁切框 padding", value=-60, minimum=-300, maximum=300, step=1)
                repair_latex = gr.Checkbox(value=True, label="修復 latex")
                full_border = gr.Checkbox(label="修復 latex 表格全框線")
                unsqueeze = gr.Checkbox(label="修復 latex 並解開多行/列合併")

        submit_button = gr.Button("生成")
        text_output = gr.Textbox(label="生成的文字輸出")

        with gr.Row():
            with gr.Column():
                crop_table_result = gr.Image(label="偵測表格結果")

            with gr.Column():
                html_output = gr.HTML(label="生成的表格輸出")

        # Constant augments
        _device_map = gr.Textbox(value=device_map, visible=False)

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
            outputs=[text_output, html_output, crop_table_result],
        )
        demo.launch(
            server_name=host,
            server_port=port,
        )


if __name__ == "__main__":
    args = arg_parser()
    args_dict = vars(args)
    test_website(**args_dict)
