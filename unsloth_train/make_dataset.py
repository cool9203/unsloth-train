# coding: utf-8

import random
from os import PathLike
from pathlib import Path
from typing import List

import datasets
import pandas as pd
from datasets import Dataset


def make_from_qa(
    dataset_path: PathLike,
    question_headers: List[str] = ["question", "q", "query"],
    answer_headers: List[str] = ["answer", "a", "response"],
    **kwds,
):
    dataset_path = Path(dataset_path)
    if dataset_path.suffix == ".csv":
        origin_dataset = datasets.load_dataset("csv", data_files=str(dataset_path), split="train")
    elif dataset_path.suffix == ".json":
        origin_dataset = datasets.load_dataset("json", data_files=str(dataset_path), split="train")
    elif dataset_path.suffix == ".xlsx":
        origin_dataset = Dataset.from_pandas(pd.read_excel(str(dataset_path)))
    else:
        origin_dataset = datasets.load_dataset(str(dataset_path), split="train")

    # Convert to [{"role": str, "content": str}, {...}, ...]
    def _to_role_content_format(examples):
        all_conversations = list()

        # Get queries
        queries = [examples.get(question_header) for question_header in question_headers if question_header in examples]
        queries = queries + [
            examples.get(question_header.capitalize())
            for question_header in question_headers
            if question_header.capitalize() in examples
        ]

        # Get responses
        responses = [examples.get(answer_header) for answer_header in answer_headers if answer_header in examples]
        responses = responses + [
            examples.get(answer_header.capitalize()) for answer_header in answer_headers if answer_header.capitalize() in examples
        ]

        # Check queries and responses
        assert queries, f"dataset not have {question_headers} key"
        assert responses, f"dataset not have {answer_headers} key"
        (queries, responses) = (queries[0], responses[0])

        for i in range(len(queries)):
            all_conversations.append(
                [
                    {"role": "user", "content": queries[i]},
                    {"role": "assistant", "content": responses[i]},
                ]
            )
        return {
            "conversations": all_conversations,
        }

    return origin_dataset.map(_to_role_content_format, batched=True, desc="Make dataset from qa")


def make_from_qa_format_3(
    dataset_path: PathLike,
    question_headers: List[str] = ["question", "q", "query"],
    answer_headers: List[str] = ["answer", "a", "response"],
    max_document_length: int = None,
    seed: int = 3407,
    **kwds,
):
    dataset_path = Path(dataset_path)
    if dataset_path.suffix == ".csv":
        origin_dataset = datasets.load_dataset("csv", data_files=str(dataset_path), split="train")
    elif dataset_path.suffix == ".json":
        origin_dataset = datasets.load_dataset("json", data_files=str(dataset_path), split="train")
    elif dataset_path.suffix == ".xlsx":
        origin_dataset = Dataset.from_pandas(pd.read_excel(str(dataset_path)))
    else:
        origin_dataset = datasets.load_dataset(str(dataset_path), split="train")

    # Convert to [{"role": str, "content": str}, {...}, ...]
    def _to_role_content_format(examples):
        all_conversations = list()
        rng = random.Random(seed)

        # Get queries
        queries = [examples.get(question_header) for question_header in question_headers if question_header in examples]
        queries = queries + [
            examples.get(question_header.capitalize())
            for question_header in question_headers
            if question_header.capitalize() in examples
        ]

        # Get responses
        responses = [examples.get(answer_header) for answer_header in answer_headers if answer_header in examples]
        responses = responses + [
            examples.get(answer_header.capitalize()) for answer_header in answer_headers if answer_header.capitalize() in examples
        ]

        # Check queries and responses
        assert queries, f"dataset not have {question_headers} key"
        assert responses, f"dataset not have {answer_headers} key"
        (queries, responses) = (queries[0], responses[0])

        def _format_qa(query: str, response: str) -> str:
            return f"Q: {query} A: {response}"

        for i in range(len(queries)):
            documents = list()
            _max_document_length = max_document_length if max_document_length else rng.randint(1, 10)
            add_current_document = False
            while len(documents) < _max_document_length:
                # Add current document
                if not add_current_document and rng.randint(1, 10) <= 5:
                    documents.append(_format_qa(query=queries[i], response=responses[i]))
                    add_current_document = True

                # Random select qa
                index = rng.randint(0, len(queries) - 1)
                if index == i:
                    continue
                documents.append(_format_qa(query=queries[index], response=responses[index]))

            if not add_current_document:
                documents.append(_format_qa(query=queries[i], response=responses[i]))

            all_conversations.append(
                [
                    {"role": "system", "content": "從<Document>裡找到答案並利用該答案回答<Query>所敘述的問題"},
                    {"role": "system", "content": "<Document>\n" + "\n\n".join(documents) + "\n</Document>"},
                    {"role": "user", "content": f"<Query>\n{queries[i]}\n</Query>"},
                    {"role": "assistant", "content": responses[i]},
                ]
            )
        return {
            "conversations": all_conversations,
        }

    return origin_dataset.map(_to_role_content_format, batched=True, desc="Make dataset from qa")
