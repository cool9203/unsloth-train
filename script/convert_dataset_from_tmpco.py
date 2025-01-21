# coding: utf-8
"""
該腳本可以轉換 `沛波鋼鐵` 鋼材資料集

Maintainer: yoga(ychsu@iii.org.tw)
Date: 2024/12/12

Dependencies package:
    - pandas
    - openpyxl
    - pillow
    - tqdm

Output: .xlsx

Folder layout example:
- root
|- orig(can be set with -f <folder name>)
|- ...
"""

import argparse
import logging
import pprint
from os import PathLike
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
)

import pandas as pd
import pypandoc
import tqdm as TQDM
from PIL import Image

logger = logging.getLogger(__name__)


def arg_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="該腳本可以轉換 沛波鋼鐵 鋼材資料集")

    parser.add_argument("-r", "--root_path", type=str, required=True, help="Folder data path")
    parser.add_argument("-o", "--output_path", type=str, required=True, help="Output data path")
    parser.add_argument("--folder_name", type=str, default=None, help="Check folder name")
    parser.add_argument("--format", type=str, choices=["latex", "markdown", "str"], required=True, help="Output text format")
    parser.add_argument("--prompt", type=str, required=True, help="After image prompt text")
    parser.add_argument("--output_name", type=str, default="data", help="Output filename")
    parser.add_argument("--copy_image", action="store_true", help="Output filename")

    parser.add_argument("--tqdm", action="store_true", help="Show progress bar")

    parser.add_argument("-v", "--verbose", action="store_true", help="Show detail")

    args = parser.parse_args()

    return args


def check_iterate_num(
    path: PathLike,
    check_name: str,
) -> int:
    iterate_num = 0
    is_exist = False
    while True:
        if Path(path, check_name).exists():
            is_exist = True
            break
        elif Path(path).is_file():
            break
        path = [p for p in Path(path).iterdir()][0]
        iterate_num += 1
    return iterate_num if is_exist else None


def load_label_data(
    iterate_num: int,
    path_chain: list[PathLike],
    folder_name: str = None,
) -> List[Dict[str, Any]]:
    root_path = "/".join([str(p) for p in path_chain])
    data = list()

    if len(path_chain) == iterate_num:
        read_filenames = set()
        path = Path(root_path, folder_name) if folder_name else Path(root_path)
        for filepath in path.glob(r"*.txt"):
            if filepath.stem not in read_filenames:
                base_name = filepath.stem
                extension_name = None
                logger.info(f"Read {base_name!s}")
                read_filenames.add(base_name)

                # Read image and check label file
                if Path(path, f"{base_name}.txt").exists():
                    if Path(path, f"{base_name}.jpg").exists():
                        extension_name = ".jpg"
                    elif Path(path, f"{base_name}.png").exists():
                        extension_name = ".png"
                    else:
                        raise FileNotFoundError("Not found image file")

                    with Path(path, f"{base_name}.txt").open("r", encoding="utf-8") as f:
                        label_content = f.read()
                else:
                    raise FileNotFoundError("Not found label file")

                data.append(
                    {
                        "label": label_content,
                        "source": base_name,
                        "full_source": str(Path(path, base_name)),
                        "image_path": str(Path(path, f"{base_name}{extension_name}")),
                    }
                )

        return data

    for path in Path(*path_chain).iterdir():
        if path.is_file():
            continue
        data += load_label_data(
            iterate_num=iterate_num,
            path_chain=path_chain + [path.name],
            folder_name=folder_name,
        )
    return data


def convert_dataset_from_tmpco(
    root_path: PathLike,
    output_path: PathLike,
    output_name: str,
    folder_name: str,
    format: str,
    prompt: str,
    copy_image: bool = False,
    tqdm: bool = False,
) -> pd.DataFrame:
    # Pre-check
    root_path = Path(root_path)
    output_path = Path(output_path)
    assert root_path.exists(), f"'{root_path!s}' not exist"

    if output_path.exists() and output_path.is_file():
        input(f"Warning: '{output_path!s}' is exists, will remove, if want press Enter...")
        output_path.unlink()

    iterate_num = (
        check_iterate_num(
            path=root_path,
            check_name=folder_name,
        )
        if folder_name
        else 0
    )
    assert iterate_num is not None, f"Error: this '{root_path!s}' can't parse with tmpco folder layout"
    logger.info(f"Iterate count: {iterate_num}")

    data = load_label_data(
        iterate_num=iterate_num + 1,
        path_chain=[root_path],
        folder_name=folder_name,
    )

    logger.info(pprint.pformat(data[:3]))

    df = pd.DataFrame(data)

    # Create image path
    logger.info("Create image path")
    image_save_path = Path(output_path, "image")
    image_save_path.mkdir(parents=True, exist_ok=True)

    iter_length = TQDM.tqdm(range(len(df))) if tqdm else range(len(df))

    # Pre-check output text format
    logger.info("Pre-check output text format")
    for i in iter_length:
        if format in ["latex", "markdown"]:
            text = pypandoc.convert_text(df.iloc[i]["label"], to="html", format=format)
            if "<table>" not in text:
                raise ValueError(f"format error: incorrect {format}")
        elif format in ["str"]:
            pass
        else:
            raise ValueError("format error")

    # Convert to dataset format
    dataset = list()
    logger.info("Convert to dataset format start")
    iter_length = TQDM.tqdm(range(len(df))) if tqdm else range(len(df))
    for i in iter_length:
        dataset.append(
            {
                "messages": [
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "text",
                                "text": "You should follow the instructions carefully and explain your answers in detail.",
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": str(Path(f'{df.iloc[i]["source"]}.jpg'))},
                            {"type": "text", "text": prompt},
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": df.iloc[i]["label"]},
                        ],
                    },
                ],
                "source": df.iloc[i]["source"],
                "full_source": df.iloc[i]["full_source"],
            }
        )
        if copy_image:
            with Image.open(df.iloc[i]["image_path"]) as image_file:
                image_file.save(str(Path(image_save_path, f'{df.iloc[i]["source"]}.jpg')))
    logger.info("Convert to dataset format end")

    dataset_save_path = Path(output_path, Path(output_name).stem + ".xlsx")
    dataset = pd.DataFrame(dataset)
    logger.info("Dataset to pd.dataframe")
    dataset.to_excel(str(dataset_save_path))
    logger.info(f"Save dataset to {dataset_save_path!s}")
    return dataset


if __name__ == "__main__":
    args = arg_parser()

    # Set logger
    logging.basicConfig(
        format="%(levelname)s %(asctime)s: %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
        level=logging.INFO if args.verbose else logging.WARNING,
    )
    delattr(args, "verbose")

    convert_dataset_from_tmpco(**vars(args))
