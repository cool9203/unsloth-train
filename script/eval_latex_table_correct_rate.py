# coding: utf-8

import argparse
import logging
import os
import pprint
import time
from dataclasses import dataclass, field
from os import PathLike
from pathlib import Path

import pandas as pd
import tqdm as TQDM
from gradio_client import Client, handle_file

from unsloth_train import utils

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(os.environ.get("LOG_LEVEL", "INFO"))


def arg_parser() -> argparse.Namespace:
    """取得執行程式時傳遞的參數

    tutorial: https://docs.python.org/zh-tw/3/howto/argparse.html#
    reference: https://docs.python.org/zh-tw/3/library/argparse.html#nargs

    Returns:
        argparse.Namespace: 使用args.name取得傳遞的參數
    """

    parser = argparse.ArgumentParser(description="Evaluation latex table model")
    parser.add_argument("--datasets", type=str, nargs="+", required=True, default=[], help="Evaluation dataset path")
    parser.add_argument("--api_url", type=str, required=True, help="Latex table model gradio api url")
    parser.add_argument("--max_tokens", type=int, default=4096, help="Model max tokens")
    parser.add_argument("--prompt", type=str, default="OCR with format:", help="Model prompt")
    parser.add_argument(
        "--system_prompt",
        type=str,
        default="You should follow the instructions carefully and explain your answers in detail.",
        help="Model system prompt",
    )
    parser.add_argument("--output", type=str, default="eval_result", help="Eval detail output path")

    args = parser.parse_args()

    return args


@dataclass
class EvalResult:
    predict_latex_error: bool = False
    gold_latex_error: bool = False
    cell_correct_count: int = 0
    cell_count: int = 0
    txt_filepath: str = None
    image_filepath: str = None
    gold_df: pd.DataFrame = None
    predict_df: pd.DataFrame = None
    error_indexes: list[tuple[str, int]] = field(default_factory=list)


def _inference_latex_table(
    client: Client,
    prompt: str,
    image_path: str,
    model_name: str,
    detect_table: bool,
    crop_table_padding: int,
    system_prompt: str,
    max_tokens: int,
    retry: int,
) -> str:
    _error = RuntimeError("inference latex table error")
    _request_data = {
        "image": handle_file(image_path),
    }
    _request_data.update(prompt=prompt) if prompt is not None else None
    _request_data.update(model_name=model_name) if model_name is not None else None
    _request_data.update(detect_table=detect_table) if detect_table is not None else None
    _request_data.update(crop_table_padding=crop_table_padding) if crop_table_padding is not None else None
    _request_data.update(system_prompt=system_prompt) if system_prompt is not None else None
    _request_data.update(max_tokens=max_tokens) if max_tokens is not None else None

    for _ in range(retry):
        try:
            response = client.predict(
                **_request_data,
                api_name="/inference_table",
            )
            return str(response[0])
        except Exception as e:
            _error = e
            time.sleep(10)
    raise _error


def calc_correct_rate(
    results: list[EvalResult],
) -> tuple[
    float,
    float,
    float,
    float,
]:
    if len(results) == 0:
        return (0.0, 0.0, 0.0, 0.0)
    return (
        float(
            f"{sum([1 if not result.predict_latex_error and not result.gold_latex_error and result.cell_correct_count == result.cell_count else 0 for result in results]) / len(results):.3}"
        ),  # Table correct rate
        float(
            f"{sum([result.cell_correct_count  for result in results]) / sum([result.cell_count  for result in results]):.3}"
        ),  # Cell correct rate
        float(
            f"{sum([1 if result.predict_latex_error else 0 for result in results]) / len(results):.3}"
        ),  # Format incorrect rate
        float(
            f"{sum([1 if result.gold_latex_error else 0 for result in results]) / len(results):.3}"
        ),  # Label format incorrect rate
    )


def eval_latex_table(
    api_url: str,
    dataset_path: PathLike,
    prompt: str = None,
    model_name: str = None,
    detect_table: bool = True,
    crop_table_padding: int = -60,
    system_prompt: str = None,
    max_tokens: int = 4096,
    retry: int = 3,
    tqdm: bool = True,
) -> tuple[list[EvalResult], float, float, float]:
    data: list[tuple[PathLike, PathLike]] = list()
    results: list[EvalResult] = list()

    # Pre-check dataset are correct pairs for label txt data and image data
    for txt_filepath in Path(dataset_path).glob("*.txt"):
        _data = None
        for image_extension in ["jpg", "png"]:
            if Path(txt_filepath.parent, f"{txt_filepath.stem}.{image_extension.lower()}").exists():
                _data = (txt_filepath, Path(txt_filepath.parent, f"{txt_filepath.stem}.{image_extension.lower()}"))
            elif Path(txt_filepath.parent, f"{txt_filepath.stem}.{image_extension.upper()}").exists():
                _data = (txt_filepath, Path(txt_filepath.parent, f"{txt_filepath.stem}.{image_extension.upper()}"))
        if _data:
            data.append(_data)
        else:
            raise ValueError(f"Not have image data: {txt_filepath!s}")
    logger.debug(f"data: {data}")

    # Connection model from gradio api
    client = Client(api_url, httpx_kwargs={"timeout": 90})

    # Eval
    for txt_filepath, image_filepath in TQDM.tqdm(data, desc="Eval") if tqdm else data:
        latex_table_text = _inference_latex_table(
            client=client,
            prompt=prompt,
            image_path=image_filepath,
            model_name=model_name,
            detect_table=detect_table,
            crop_table_padding=crop_table_padding,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            retry=retry,
        )

        (gold_df, predict_df) = (None, None)
        try:
            with Path(txt_filepath).open(mode="r", encoding="utf-8") as f:
                gold_df = utils.convert_latex_table_to_pandas(latex_table_str=f.read(), headers=True)
            predict_df = utils.convert_latex_table_to_pandas(latex_table_str=latex_table_text, headers=True)
        except Exception as e:
            logger.exception(e)

        logger.debug(gold_df)
        logger.debug("-" * 25)
        logger.debug(predict_df)
        logger.debug("-" * 25)

        result = EvalResult(
            txt_filepath=str(txt_filepath),
            image_filepath=str(image_filepath),
            gold_df=gold_df,
            predict_df=predict_df,
        )

        if predict_df is None:
            result.predict_latex_error = True
        elif gold_df is None:
            result.gold_latex_error = True

        else:
            result.cell_count = len(gold_df.columns) * len(gold_df) + len(gold_df.columns)

            # Compare header(column)
            for column_index in range(min(len(gold_df.columns), len(predict_df.columns))):
                gold_text = gold_df.columns[column_index].replace(" ", "").replace("\n", "").replace(":", "")
                predict_text = predict_df.columns[column_index].replace(" ", "").replace("\n", "").replace(":", "")
                if gold_text == predict_text:
                    result.cell_correct_count += 1
                else:
                    result.error_indexes.append((column_index, None))

            # Compare row
            for column_index in range(min(len(gold_df.columns), len(predict_df.columns))):
                for row_index in range(min(len(gold_df), len(predict_df))):
                    gold_text = gold_df.iloc[row_index, column_index].replace(" ", "").replace("\n", "").replace(":", "")
                    predict_text = predict_df.iloc[row_index, column_index].replace(" ", "").replace("\n", "").replace(":", "")
                    if gold_text == predict_text:
                        result.cell_correct_count += 1
                    else:
                        result.error_indexes.append((column_index, row_index))
        results.append(result)

    return tuple(
        [
            results,
            *calc_correct_rate(results=results),
        ]
    )


if __name__ == "__main__":
    args = arg_parser()
    common_parameters = vars(args)

    logger.info(f"Used parameters:\n{pprint.pformat(common_parameters)}")

    dataset_paths = common_parameters.pop("datasets")
    output_path = common_parameters.pop("output")

    results = []
    for dataset_path in dataset_paths:
        dataset_path_split = dataset_path.split(":", maxsplit=1)
        passed_parameters = common_parameters.copy()
        for parameter in dataset_path_split[1].replace(" ", "").split(","):
            parameter_split = parameter.split("=", maxsplit=1)
            passed_parameters.update({parameter_split[0]: parameter_split[1]})

        results += eval_latex_table(
            dataset_path=dataset_path_split[0],
            **passed_parameters,
        )[0]

    correct_rate = calc_correct_rate(results=results)
    logger.info(f"Table correct rate: {correct_rate[0]}")
    logger.info(f"Cell correct rate: {correct_rate[1]}")
    logger.info(f"Format incorrect rate: {correct_rate[2]}")
    logger.info(f"Label format incorrect rate: {correct_rate[3]}")

    with Path(
        Path(output_path).parent,
        f"{Path(output_path).stem}.txt",
    ).open("w", encoding="utf-8") as f:
        f.write(
            ("-" * 25 + "\n").join(
                [
                    pprint.pformat(result).replace("gold_df=  ", "gold_df=\n").replace("predict_df=  ", "predict_df=\n")
                    for result in results
                ]
            )
            + "-" * 25
            + "\n\n"
        )
        f.writelines(
            [
                f"Table correct rate: {correct_rate[0]}\n",
                f"Cell correct rate: {correct_rate[1]}\n",
                f"Format incorrect rate: {correct_rate[2]}\n",
                f"Label format incorrect rate: {correct_rate[3]}\n",
            ]
        )
