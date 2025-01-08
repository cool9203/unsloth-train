# coding: utf-8

import argparse
import re

import gradio as gr
import pypandoc
from transformers import MllamaForConditionalGeneration, MllamaProcessor
from unsloth import FastVisionModel

_latex_tabular_pattern = r"(\\begin[\S\s]*\\end{tabular})"
_markdown_table_pattern = r"(\|[\S\s]*\|)"

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
    parser.add_argument("--model_name_or_path", type=str, default=None, help="Run model name or path")
    parser.add_argument("--max_new_tokens", type=int, default=4096, help="Run model generate max new tokens")

    args = parser.parse_args()

    return args


def load_model(
    model_name_or_path: str,
    load_in_4bit: bool = True,
):
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=model_name_or_path,  # YOUR MODEL YOU USED FOR TRAINING
        load_in_4bit=load_in_4bit,  # Set to False for 16bit LoRA
    )
    FastVisionModel.for_inference(model)  # Enable for inference!
    return (model, tokenizer)


def generate(
    image,
    prompt: str,
    device_map: str = "auto",
    max_new_tokens: int = 1024,
    **kwds,
) -> str:
    tokenizer = __model.get("tokenizer")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "You should follow the instructions carefully and explain your answers in detail."}
            ],
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
    output = __model.get("model").generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        **kwds,
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)


def handle_request(
    model_name_or_path: str,
    image,
    prompt: str,
    device_map: str = "auto",
    max_new_tokens: int = 4096,
    use_cache: bool = True,
    top_p: float = 1.0,
):
    if model_name_or_path != __model.get("name", None):
        (__model["model"], __model["tokenizer"]) = load_model(model_name_or_path=model_name_or_path)
        __model["name"] = model_name_or_path
    origin_response = generate(
        prompt=prompt,
        image=image,
        device_map=device_map,
        max_new_tokens=max_new_tokens,
        use_cache=use_cache,
        top_p=top_p,
        do_sample=False,
    )
    try:
        origin_response = re.findall(_latex_tabular_pattern, origin_response)[0]
        html_response = pypandoc.convert_text(origin_response, "html", format="latex")
    except Exception as e:
        try:
            origin_response = re.findall(_markdown_table_pattern, origin_response)[0]
            html_response = pypandoc.convert_text(origin_response, "html", format="markdown")
        except Exception as e:
            html_response = "輸出的內容不是正確的 latex or markdown"
    return origin_response, html_response


def test_website(
    host: str = "127.0.0.1",
    port: int = 7860,
    model_name_or_path: str = None,
    max_new_tokens: int = 4096,
    **kwds,
):
    if model_name_or_path and __model.get("name") is None:
        (__model["model"], __model["tokenizer"]) = load_model(model_name_or_path=model_name_or_path)
        __model["name"] = model_name_or_path

    # Gradio 接口定義
    with gr.Blocks() as demo:
        gr.Markdown("## LLM 測試網站（支持圖片與文本輸入）")

        with gr.Row():
            with gr.Column():
                _model_name_or_path = gr.Textbox(
                    label="模型名稱或路徑", value=__model.get("name", None), visible=not model_name_or_path
                )
                image_input = gr.Image(label="上傳圖片", type="pil")
                prompt_input = gr.Textbox(label="輸入文字提示", lines=2, value="latex table ocr")
                submit_button = gr.Button("生成")

            with gr.Column():
                html_output = gr.HTML(label="生成的表格輸出")
                text_output = gr.Textbox(label="生成的文字輸出")

        # Constant augments
        _device_map = gr.Textbox(value="cuda:0", visible=False)
        _max_new_tokens = gr.Number(value=max_new_tokens, visible=False)

        submit_button.click(
            handle_request,
            inputs=[
                _model_name_or_path,
                image_input,
                prompt_input,
                _device_map,
                _max_new_tokens,
            ],
            outputs=[text_output, html_output],
        )
        demo.launch(
            server_name=host,
            server_port=port,
        )


if __name__ == "__main__":
    args = arg_parser()
    args_dict = vars(args)
    test_website(**args_dict)
