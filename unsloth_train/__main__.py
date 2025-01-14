# coding: utf-8

import argparse
import pprint

from unsloth_train.utils import _get_function_used_params


def arg_parser() -> argparse.Namespace:
    """取得執行程式時傳遞的參數

    tutorial: https://docs.python.org/zh-tw/3/howto/argparse.html#
    reference: https://docs.python.org/zh-tw/3/library/argparse.html#nargs

    Returns:
        argparse.Namespace: 使用args.name取得傳遞的參數
    """

    parser = argparse.ArgumentParser(description="Run unsloth training with llm or vlm")
    parser.add_argument(
        "script_name",
        type=str,
        choices=[
            "text",
            "vision",
            "test_web",
        ],
        help="Run training script name",
    )

    # Common
    parser.add_argument("-o", "--output_model_path", type=str, default=None, help="Save model path")
    parser.add_argument(
        "-m",
        "--model_name",
        type=str,
        default=None,
        help="Training model name, need full, like 'unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit'",
    )
    parser.add_argument("-d", "--dataset_path", type=str, default=None, help="Training dataset path")
    parser.add_argument("--seed", type=int, default=3407, help="Random seed")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Train data max sequence length")
    parser.add_argument("--load_in_4bit", action="store_true", help="Model load in 4bit with bitsandbytes")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Train model epochs")
    parser.add_argument("--learning_rate", type=float, default=7e-6, help="Train model learning rate")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoint", help="Train model checkpoint save path")
    parser.add_argument(
        "--target_modules",
        type=str,
        nargs="+",
        default=[],
        help="Train model target modules",
    )

    # VLM only
    parser.add_argument(
        "--finetune_vision_layers",
        action="store_true",
        help="\033[32;1;4m(VLM only)\033[0m Model train vision layers",
    )
    parser.add_argument(
        "--finetune_language_layers",
        action="store_true",
        help="\033[32;1;4m(VLM only)\033[0m Model train language layers",
    )
    parser.add_argument(
        "--finetune_attention_modules",
        action="store_true",
        help="\033[32;1;4m(VLM only)\033[0m Model train attention layers",
    )
    parser.add_argument(
        "--finetune_mlp_modules",
        action="store_true",
        help="\033[32;1;4m(VLM only)\033[0m Model train mlp layers",
    )

    # LLM only
    parser.add_argument(
        "--quantization_method",
        type=str,
        nargs="+",
        default=["q4_k_m"],
        help="\033[31;1;4m(LLM only)\033[0m Model save quantization method",
    )
    parser.add_argument(
        "--chat_template_name",
        type=str,
        default=None,
        help="\033[31;1;4m(LLM only)\033[0m Model chat template name",
    )

    # Dataset only
    parser.add_argument(
        "--max_document_length",
        type=int,
        default=5,
        help="\033[33;1;4m(Dataset parameter)\033[0m Make from qa max document length",
    )
    parser.add_argument(
        "--not_answering_proportion",
        type=float,
        default=1.0,
        help="\033[33;1;4m(Dataset parameter)\033[0m Make from qa not answering proportion",
    )
    parser.add_argument(
        "--bm25",
        action="store_true",
        help="\033[33;1;4m(Dataset parameter)\033[0m Make from qa document search method use bm25",
    )
    parser.add_argument(
        "--qa_format",
        type=int,
        default=-1,
        help="\033[33;1;4m(Dataset parameter)\033[0m Make from qa document format",
    )
    parser.add_argument(
        "--dataset_text_field",
        type=str,
        default="messages",
        help="\033[33;1;4m(Dataset parameter)\033[0m Make dataset text field name",
    )
    parser.add_argument(
        "--image_root_path",
        type=str,
        default="",
        help="\033[33;1;4m(Dataset parameter)\033[0m Dataset image path common root path",
    )

    # Test website only
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="\033[35;1;4m(Website parameter)\033[0m Web server host",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="\033[35;1;4m(Website parameter)\033[0m Web server port",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        help="\033[35;1;4m(Website parameter)\033[0m Run model name or path",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=4096,
        help="\033[35;1;4m(Website parameter)\033[0m Run model generate max new tokens",
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default="cuda:0",
        help="\033[35;1;4m(Website parameter)\033[0m Run model device map",
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    check_arguments = [
        "output_model_path",
        "model_name",
        "dataset_path",
    ]
    make_dataset_parameter_keys = {
        "max_document_length",
        "not_answering_proportion",
        "bm25",
        "dataset_text_field",
        "image_root_path",
    }
    args = arg_parser()

    # Pre-process arguments
    for check_argument in check_arguments:
        if not getattr(args, check_argument):
            delattr(args, check_argument)

    args_dict = vars(args)

    # Convert dataset parameters
    import unsloth_train.make_dataset

    if args.qa_format >= 0:
        make_dataset_fn = getattr(unsloth_train.make_dataset, f"make_from_qa_format_{args.qa_format}", None)
        assert make_dataset_fn, f"Not support this qa_format: '{args.qa_format}'"
    else:
        make_dataset_fn = getattr(unsloth_train.make_dataset, "make_from_universal")
    args_dict["make_dataset_fn"] = make_dataset_fn

    make_dataset_parameters = dict()
    for key in make_dataset_parameter_keys:
        make_dataset_parameters.update({key: args_dict[key]})
        del args_dict[key]
    args_dict["make_dataset_parameters"] = make_dataset_parameters

    # Check and convert target_modules
    if not args.target_modules:
        del args_dict["target_modules"]

    print(pprint.pformat(args_dict))

    if args.script_name == "text":
        from unsloth_train.train import train_model

        parameters = _get_function_used_params(train_model, **args_dict)
        train_model(**parameters)
    elif args.script_name == "vision":
        from unsloth_train.train_vision import train_model

        parameters = _get_function_used_params(train_model, **args_dict)
        train_model(**parameters)
    elif args.script_name == "test_web":
        from unsloth_train.test_website import test_website

        parameters = _get_function_used_params(test_website, **args_dict)
        test_website(**parameters)
