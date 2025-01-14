# coding: utf-8

import ast
import hashlib
import random
import re
from os import PathLike
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Sequence,
    Tuple,
    Union,
)

import datasets
import pandas as pd
import tqdm
from datasets import Dataset
from PIL import Image as PILImage

_support_image_format = {ex for ex, f in PILImage.registered_extensions().items() if f in PILImage.OPEN}
_bm25_model_cache = dict()


def check_image_path(
    image_path: PathLike,
    image_root_path: PathLike = "",
) -> Union[str, None]:
    image_path = Path(image_root_path, image_path) if image_root_path and Path(image_root_path).exists() else Path(image_path)
    if image_path.suffix in _support_image_format:  # Is image
        if image_path.exists():
            return image_path
        else:
            FileNotFoundError(f"You seem pass an image, but not exist. path: '{str(image_path)}'")
    return None


def load_dataset(
    dataset_path: PathLike,
) -> pd.DataFrame:
    dataset_path = Path(dataset_path)
    suffix = dataset_path.suffix[1:]
    if suffix == "xlsx":
        dataset = pd.read_excel(str(dataset_path))
    else:
        load_fn = getattr(pd, f"read_{suffix}", None)
        assert load_fn, f"Not support {suffix} file type"
        dataset = load_fn(str(dataset_path))
    return dataset


def bm25_search(
    corpus: List[str],
    query: str,
    stop_word: Sequence = {" "},
    k: int = None,
    threshold: float = None,
) -> List[Tuple[int, str, float]]:
    import jieba
    from gensim.corpora import Dictionary
    from gensim.models import OkapiBM25Model, TfidfModel
    from gensim.similarities import SparseMatrixSimilarity

    bm25_result: List[Tuple[int, str, float]] = list()
    corpus_hash_value = hashlib.sha256(("\n".join(corpus)).encode())

    if corpus_hash_value in _bm25_model_cache:
        _corpus = _bm25_model_cache[corpus_hash_value]["corpus"]
        dictionary = _bm25_model_cache[corpus_hash_value]["dictionary"]
        query_model = _bm25_model_cache[corpus_hash_value]["query_model"]
        document_model = _bm25_model_cache[corpus_hash_value]["document_model"]
        bow_corpus = _bm25_model_cache[corpus_hash_value]["bow_corpus"]
        bm25_corpus = _bm25_model_cache[corpus_hash_value]["bm25_corpus"]
    else:
        # Word segment with jieba
        _corpus = list()
        for sentence in corpus:
            _corpus.append([w for w in jieba.cut(sentence) if w not in stop_word])

        # Create model
        dictionary = Dictionary(_corpus)  # fit dictionary
        query_model = TfidfModel(dictionary=dictionary, smartirs="bnn")  # enforce binary weights
        document_model = OkapiBM25Model(dictionary=dictionary)  # fit bm25 model
        bow_corpus = [dictionary.doc2bow(line) for line in _corpus]  # convert corpus to BoW format
        bm25_corpus = document_model[bow_corpus]
        _bm25_model_cache[corpus_hash_value] = {
            "corpus": _corpus,
            "dictionary": dictionary,
            "query_model": query_model,
            "document_model": document_model,
            "bow_corpus": bow_corpus,
            "bm25_corpus": bm25_corpus,
        }

    # Querying with bm25
    index = SparseMatrixSimilarity(
        bm25_corpus, num_docs=len(_corpus), num_terms=len(dictionary), normalize_queries=False, normalize_documents=False
    )
    query_segment = [w for w in jieba.cut(query) if w not in stop_word]
    bow_query = dictionary.doc2bow(query_segment)
    bm25_query = query_model[bow_query]
    scores = index[bm25_query]  # calculate similarity of query to each doc from bow_corpus
    sorted_scores = sorted([(i, 1 / s if s > 0 else 1.0) for i, s in enumerate(scores)], key=lambda tup: tup[1])  # small to large
    for i, score in sorted_scores:
        if (k is not None and len(bm25_result) == k) or (threshold is not None and score > threshold):
            break

        bm25_result.append((i, corpus[i], score))

    return bm25_result


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
    bm25: bool = False,
    extra_bm25_result: bool = False,
    **kwds,
):
    report_once_status = False
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
    for i, is_positive in tqdm.tqdm(positive_indexes + negative_indexes):
        question = origin_dataset.iloc[i][question_header]
        answer = origin_dataset.iloc[i][answer_header]
        new_dataset["question"].append(question)
        new_dataset["answer"].append(answer if is_positive else not_answering_response)
        new_dataset["reflection"].append("")

        references = list()
        _max_document_length = max_document_length if max_document_length else rng.randint(1, 10)

        bm25_results: List[Tuple[int, str, float]] = (
            bm25_search(
                corpus=origin_dataset[question_header].tolist(),
                query=question,
            )
            if bm25
            else []
        )

        if is_positive:
            references.append(_format_qa(query=question, response=answer))

        if bm25 and not extra_bm25_result:
            print("Run bm25") if not report_once_status else None
            for index, _, score in bm25_results:
                if len(references) >= _max_document_length:
                    break
                if index == i or (not is_positive and not score == 1.0):
                    continue
                references.append(
                    _format_qa(
                        query=origin_dataset.iloc[index][question_header],
                        response=origin_dataset.iloc[index][answer_header],
                    )
                )
        else:
            print("Run random") if not report_once_status else None
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

        if bm25 and extra_bm25_result:
            print("Run extra-bm25") if not report_once_status else None
            references = list()
            new_dataset["question"].append(question)
            new_dataset["answer"].append(answer if is_positive else not_answering_response)
            new_dataset["reflection"].append("")
            for index, _, score in bm25_results:
                if len(references) >= _max_document_length:
                    break
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
        report_once_status = True

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


def make_from_universal(
    dataset_path: PathLike,
    seed: int = 3407,
    dataset_text_field: str = "messages",
    image_root_path: str = "",
):
    origin_dataset = load_dataset(dataset_path)
    assert dataset_text_field in origin_dataset.columns, f"'{dataset_text_field}' key must in dataset"
    dataset = Dataset.from_pandas(df=pd.DataFrame(origin_dataset))

    # Convert to [{"role": str, "content": str}, {...}, ...]
    def _to_role_content_format(
        sample: Union[Dict[str, Any], str],
        image_root_path: str,
    ):
        # Get entries
        if isinstance(sample, str):
            messages_str = sample
        else:
            messages_str = sample.get("messages")
        try:
            messages = ast.literal_eval(messages_str) if isinstance(messages_str, str) else messages_str
        except SyntaxError as e:
            # Fix pandas save will use [{}\n {}] to replace [{}, {}] error
            _pandas_list_dict_pattern = r"\}\n *\{"
            _fix_pandas_list_dict_pattern = r"}, {"
            if re.findall(_pandas_list_dict_pattern, messages_str):
                messages = re.sub(
                    _pandas_list_dict_pattern,
                    _fix_pandas_list_dict_pattern,
                    messages_str,
                )
                messages = ast.literal_eval(messages)
            else:
                raise e from e

        if isinstance(messages, list):
            pass
        elif isinstance(messages, str):
            messages = [{"role": "assistant", "content": messages}]
        elif isinstance(messages, dict):
            messages = [messages]
        else:
            raise ValueError(f"'{messages}' not python ast")

        # Convert image type
        for message_index in range(len(messages)):
            contents = messages[message_index]["content"]
            messages[message_index]["role"] = (
                "user" if messages[message_index]["role"] in ["system", "user"] else messages[message_index]["role"]
            )
            if isinstance(contents, str):
                image_path = check_image_path(image_path=contents, image_root_path=image_root_path)
                messages[message_index]["content"] = image_path if image_path else contents
            elif isinstance(contents, list):
                for content_index, content in enumerate(contents):
                    if "type" in content and content["type"] in ["image"]:
                        image_path = check_image_path(
                            image_path=content[content["type"]],
                            image_root_path=image_root_path,
                        )
                        if image_path:
                            messages[message_index]["content"][content_index]["image"] = image_path
                        else:
                            messages[message_index]["content"][content_index] = {
                                "type": "text",
                                "text": content[content["type"]],
                            }
            else:
                raise ValueError(f"Not implement format: '{messages}'")

        return {"messages": messages}

    dataset = dataset.shuffle(seed=seed)
    return [_to_role_content_format(sample, image_root_path=image_root_path) for sample in dataset]
