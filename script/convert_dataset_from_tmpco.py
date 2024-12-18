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
import shutil
from os import PathLike
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
)

import pandas as pd
from PIL import Image

logger = logging.getLogger(__name__)


def arg_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="該腳本可以轉換 沛波鋼鐵 鋼材資料集")

    parser.add_argument("-r", "--root_path", type=str, required=True, help="Folder data path")
    parser.add_argument("-o", "--output_path", type=str, required=True, help="Output data path")
    parser.add_argument("-f", "--folder_name", type=str, default="orig", help="Check folder name")

    parser.add_argument("-v", "--verbose", action="store_true", help="Show detail")

    args = parser.parse_args()

    return args


def check_iterate_num(
    path: PathLike,
    check_name: str = "orig",
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


def _read_label_data_and_copy_image(
    iterate_num: int,
    path_chain: list[PathLike],
) -> List[Dict[str, Any]]:
    root_path = "/".join([str(p) for p in path_chain])
    data = list()

    if len(path_chain) == iterate_num:
        read_filenames = set()
        for path in Path(root_path, "orig").glob(r"*.txt"):
            if path.stem not in read_filenames:
                base_name = path.stem
                logger.info(f"Read {base_name!s}")
                read_filenames.add(base_name)

                # Read image and check label file
                if Path(root_path, "orig", f"{base_name}.txt").exists():
                    if Path(root_path, "orig", f"{base_name}.jpg").exists():
                        image = Image.open(Path(root_path, "orig", f"{base_name}.jpg"))
                    elif Path(root_path, "orig", f"{base_name}.png").exists():
                        image = Image.open(Path(root_path, "orig", f"{base_name}.png"))
                    else:
                        raise FileNotFoundError("Not found image file")

                    with Path(root_path, "orig", f"{base_name}.txt").open("r", encoding="utf-8") as f:
                        label_content = f.read()
                else:
                    raise FileNotFoundError("Not found label file")

                data.append(
                    {
                        "label": label_content,
                        "source": base_name,
                        "full_source": str(Path(root_path, "orig", base_name)),
                        "image": image,
                    }
                )

        return data

    for path in Path(*path_chain).iterdir():
        if path.is_file():
            continue
        data += _read_label_data_and_copy_image(
            iterate_num=iterate_num,
            path_chain=path_chain + [path.name],
        )
    return data


def convert_dataset_from_tmpco(
    root_path: PathLike,
    output_path: PathLike,
    folder_name: str,
) -> pd.DataFrame:
    # Pre-check
    root_path = Path(root_path)
    output_path = Path(output_path)
    assert root_path.exists(), f"'{root_path!s}' not exist"

    if output_path.exists():
        input(f"Warning: '{output_path!s}' is exists, will remove, if want press Enter...")
        if output_path.is_dir():
            shutil.rmtree(output_path)
        else:
            output_path.unlink()

    iterate_num = check_iterate_num(
        path=root_path,
        check_name=folder_name,
    )
    assert iterate_num is not None, f"Error: this '{root_path!s}' can't parse with tmpco folder layout"
    logger.info(f"Iterate count: {iterate_num}")

    data = _read_label_data_and_copy_image(
        iterate_num=iterate_num + 1,
        path_chain=[root_path],
    )

    logger.info(pprint.pformat(data[:3]))

    df = pd.DataFrame(data)

    # Create image path
    logger.info("Create image path")
    image_save_path = Path(output_path, "image")
    image_save_path.mkdir(parents=True, exist_ok=True)

    # Convert to dataset format
    dataset = list()
    logger.info("Convert to dataset format")
    for i in range(len(df)):
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
                            {"type": "image", "image": str(Path("image", f'{df.iloc[i]["source"]}.jpg'))},
                            {"type": "text", "text": "latex table ocr"},
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
        df.iloc[i]["image"].save(str(Path(image_save_path, f'{df.iloc[i]["source"]}.jpg')))

    dataset_save_path = Path(output_path, "data.xlsx")
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
