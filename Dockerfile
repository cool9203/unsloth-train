ARG CUDA_VERSION=12.4.1

FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu20.04 AS base

LABEL MAINTAINER="ychsu@iii.org.com"

ARG CUDA_VERSION_SHORT=124
ARG PYTHON_VERSION=python3.12
ARG UV_VERSION=0.5.7

WORKDIR /app

RUN apt update && \
    apt install -y git

COPY  --from=ghcr.io/astral-sh/uv:${UV_VERSION} /uv /bin/uv
COPY ./pyproject.toml ./README.md ./
COPY ./unsloth_train/__init__.py ./unsloth_train/__init__.py

RUN uv venv -p ${PYTHON_VERSION} && \
    uv pip install -U pip setuptools hatchling editables && \
    uv pip install torch --index-url https://download.pytorch.org/whl/cu${CUDA_VERSION_SHORT}
RUN uv pip install --no-build-isolation flash-attn==v2.7.2.post1
RUN uv pip install -e .

COPY ./unsloth_train ./unsloth_train

ENTRYPOINT [ "./.venv/bin/python", "unsloth_train" ]
