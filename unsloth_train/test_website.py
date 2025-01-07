# coding: utf-8

import re
import traceback

import gradio as gr
import pypandoc
from transformers import MllamaForConditionalGeneration, MllamaProcessor
from unsloth import FastVisionModel

_latex_tabular_pattern = r"(\\begin[\S\s]*\\end{tabular})"

__model: dict[str, MllamaForConditionalGeneration | MllamaProcessor | str] = {
    "model": None,
    "tokenizer": None,
    "name": None,
}


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
    use_cache: bool = True,
    temperature: float = 0.0,
    min_p: float = 0.1,
):
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
    input_text = __model.get("tokenizer").apply_chat_template(messages, add_generation_prompt=True)
    inputs = __model.get("tokenizer")(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt",
    ).to(device_map)
    return __model.get("model").generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        use_cache=use_cache,
        temperature=temperature,
        min_p=min_p,
        do_sample=False,
    )[0]


def handle_request(
    model_name_or_path: str,
    image,
    prompt: str,
):
    if model_name_or_path != __model.get("name", None):
        (__model["model"], __model["tokenizer"]) = load_model(model_name_or_path=model_name_or_path)
        __model["name"] = model_name_or_path
    origin_response = generate(
        prompt=prompt,
        image=image,
        device_map="cuda:0",
    )
    try:
        origin_response = re.findall(_latex_tabular_pattern, origin_response)[0]
        html_response = pypandoc.convert_text(origin_response, "html", format="latex")
    except Exception as e:
        traceback.print_exception(e)
        html_response = "輸出的內容不是正確的 latex"
    return origin_response, html_response


def test_website(
    host: str = "127.0.0.1",
    port: int = 7860,
    **kwds,
):
    # Gradio 接口定義
    with gr.Blocks() as demo:
        gr.Markdown("## LLM 測試網站（支持圖片與文本輸入）")

        with gr.Row():
            with gr.Column():
                model_name_or_path = gr.Textbox(label="模型名稱或路徑", value=__model.get("name", None))
                image_input = gr.Image(label="上傳圖片", type="pil")
                prompt_input = gr.Textbox(label="輸入文字提示", lines=2, value="latex table ocr")
                submit_button = gr.Button("生成")

            with gr.Column():
                html_output = gr.HTML(label="生成的表格輸出")
                text_output = gr.Textbox(label="生成的文字輸出")

        submit_button.click(
            handle_request,
            inputs=[
                model_name_or_path,
                image_input,
                prompt_input,
            ],
            outputs=[text_output, html_output],
        )
        demo.launch(
            server_name=host,
            server_port=port,
        )


if __name__ == "__main__":
    test_website()
