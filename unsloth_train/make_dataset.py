# coding: utf-8

import random
from os import PathLike
from pathlib import Path
from typing import List

import datasets
import pandas as pd
from datasets import Dataset


def load_dataset(
    dataset_path: PathLike,
) -> pd.DataFrame:
    dataset_path = Path(dataset_path)
    suffix = dataset_path.suffix.replace(".", "", 1)
    if suffix == "xlsx":
        dataset = pd.read_excel(str(dataset_path))
    else:
        load_fn = getattr(pd, f"read_{suffix}", None)
        assert load_fn, f"Not support {suffix} file type"
        dataset = load_fn(str(dataset_path))
    return dataset


def preprocess_dataset() -> pd.DataFrame:
    pass


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


def make_from_qa_format_4(
    dataset_path: PathLike,
    question_headers: List[str] = ["question", "q", "query"],
    answer_headers: List[str] = ["answer", "a", "response"],
    max_document_length: int = None,
    not_answering_proportion: float = 0.6,
    not_answering_response: str = "很抱歉我沒有這問題的相關答案。",
    seed: int = 3407,
    **kwds,
):
    origin_dataset = load_dataset(dataset_path)
    rng = random.Random(seed)

    # Make not answering examples
    positive_indexes = [(i, True) for i in range(len(origin_dataset))]
    negative_indexes = [(i, False) for i in range(len(origin_dataset))]
    rng.shuffle(negative_indexes)
    negative_indexes = negative_indexes[: int(len(origin_dataset) * not_answering_proportion)]
    print(
        f"Make '{len(origin_dataset)} * {not_answering_proportion} = {len(negative_indexes)}' not answering examples, all have '{len(origin_dataset)+len(negative_indexes)}'"
    )

    question_header = [
        key
        for key in [
            *question_headers,
            *[k.capitalize() for k in question_headers],
        ]
        if key in origin_dataset.columns
    ]
    answer_header = [
        key
        for key in [
            *answer_headers,
            *[k.capitalize() for k in answer_headers],
        ]
        if key in origin_dataset.columns
    ]

    # Check queries and responses
    assert question_header, f"dataset not have {question_headers} key"
    assert answer_header, f"dataset not have {answer_headers} key"
    (question_header, answer_header) = (question_header[0], answer_header[0])

    def _format_qa(query: str, response: str) -> str:
        return f"Q: {query} A: {response}"

    new_dataset = {
        "question": [],
        "answer": [],
        "reference": [],
        "reflection": [],
    }
    for i, is_positive in positive_indexes + negative_indexes:
        question = origin_dataset.iloc[i][question_header]
        answer = origin_dataset.iloc[i][answer_header]
        new_dataset["question"].append(question)
        new_dataset["answer"].append(answer if is_positive else not_answering_response)
        new_dataset["reflection"].append("")

        references = list()
        _max_document_length = max_document_length if max_document_length else rng.randint(1, 10)

        if is_positive:
            references.append(_format_qa(query=question, response=answer))

        while len(references) < _max_document_length:
            # Random select qa
            index = rng.randint(0, len(origin_dataset) - 1)
            if index == i:
                continue
            references.append(
                _format_qa(
                    query=origin_dataset.iloc[index][question_header],
                    response=origin_dataset.iloc[index][answer_header],
                )
            )
        rng.shuffle(references)
        new_dataset["reference"].append(references)

    dataset = Dataset.from_pandas(df=pd.DataFrame(new_dataset))

    # Convert to [{"role": str, "content": str}, {...}, ...]
    def _to_role_content_format(examples):
        # Get entries
        queries = examples.get("question")
        answers = examples.get("answer")
        references = examples.get("reference")
        reflections = examples.get("reflection")

        all_conversations = list()
        for i in range(len(queries)):
            messages = [
                {"role": "system", "content": "從<Document>裡找到答案並利用該答案回答<Query>所敘述的問題"},
                {"role": "system", "content": "<Document>\n" + "\n\n".join(references[i]) + "\n</Document>"},
                {"role": "user", "content": f"<Query>\n{queries[i]}\n</Query>"},
            ]
            if reflections[i]:
                messages.append(
                    {
                        "role": "assistant",
                        "content": f"<Reflection>\n{reflections[i]}\n</Reflection>\n\n{answers[i]}",
                    }
                )
            else:
                messages.append({"role": "assistant", "content": answers[i]})

            all_conversations.append(messages)
        return {"conversations": all_conversations}

    dataset = dataset.shuffle(seed=seed)
    return dataset.map(_to_role_content_format, batched=True, desc="Make dataset from qa")
