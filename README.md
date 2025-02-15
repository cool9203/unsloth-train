# unsloth-train

A use unsloth to train llm model script

## From container

### Install

#### Pre-install

[nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

### Train model with docker

```bash
# Train text only llm
docker run -d \
    --gpu=all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v ./models:/app/models \
    -v ./checkpoint:/app/checkpoint \
    -v <DATASET_PATH>:/app/<DATASET_FILENAME> \
    --name unsloth-train <IMAGE_NAME> \
    text \
    --load_in_4bit \
    -o models/llama3.2-3B-Instruct \
    -m unsloth/Llama-3.2-3B-Instruct-bnb-4bit \
    -d <DATASET_FILENAME> \
    --num_train_epochs 3

# Train vision llm
docker run -d \
    --gpu=all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v ./models:/app/models \
    -v ./checkpoint:/app/checkpoint \
    -v <DATASET_PATH>:/app/<DATASET_FILENAME> \
    --name unsloth-train <IMAGE_NAME> \
    vision \
    --load_in_4bit \
    -o models/llama3.2-11B-Vision-Instruct \
    -m unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit \
    -d <DATASET_FILENAME> \
    --finetune_language_layers \
    --finetune_attention_modules \
    --finetune_mlp_modules \
    --num_train_epochs 3
```

### Train model with podman

```bash
# Train text only llm
podman run -d \
    --device=nvidia.com/gpu=all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v ./models:/app/models \
    -v ./checkpoint:/app/checkpoint \
    -v <DATASET_PATH>:/app/<DATASET_FILENAME> \
    --name unsloth-train <IMAGE_NAME> \
    text \
    --load_in_4bit \
    -o models/llama3.2-3B-Instruct \
    -m unsloth/Llama-3.2-3B-Instruct-bnb-4bit \
    -d <DATASET_FILENAME> \
    --num_train_epochs 3

# Train vision llm
podman run -d \
    --device=nvidia.com/gpu=all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v ./models:/app/models \
    -v ./checkpoint:/app/checkpoint \
    -v <DATASET_PATH>:/app/<DATASET_FILENAME> \
    --name unsloth-train <IMAGE_NAME> \
    vision \
    --load_in_4bit \
    -o models/llama3.2-11B-Vision-Instruct \
    -m unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit \
    -d <DATASET_FILENAME> \
    --finetune_language_layers \
    --finetune_attention_modules \
    --finetune_mlp_modules \
    --num_train_epochs 3
```

## From source

### Install

```bash
git clone https://github.com/cool9203/unsloth-train.git
cd unsloth-train

# If create venv
# python -m venv ./.venv
# . ./.venv/bin/activate

python -m pip install --upgrade pip setuptools editables wheel
pip install torch # Need choice your cuda version
pip install -r requirements.txt
```

### Train model

```bash
# Train text only llm
python unsloth_train text \
    --load_in_4bit \
    -o models/llama3.2-3B-Instruct \
    -m unsloth/Llama-3.2-3B-Instruct-bnb-4bit \
    -d <DATASET_FILENAME> \
    --num_train_epochs 3

# Train vision llm
python unsloth_train vision \
    --load_in_4bit \
    -o models/llama3.2-11B-Vision-Instruct \
    -m unsloth/Llama-3.2-11B-Vision-Instruct-bnb-4bit \
    -d <DATASET_FILENAME> \
    --finetune_language_layers \
    --finetune_attention_modules \
    --finetune_mlp_modules \
    --num_train_epochs 3
```

## Prepare train data

Accept pandas can read file.

Data format:

| index | messages                                                                                                                                                                                                                                                                                                                                           |
| :---: | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
|   0   | [{'role': 'system', 'content': [{'type': 'text', 'text': 'You should follow the instructions carefully and explain your answers in detail.'}]}, {'role': 'user', 'content': [{'type': 'image', 'image': 'image path'}, {'type': 'text', 'text': 'latex table ocr'}]}, {'role': 'assistant', 'content': [{'type': 'text', 'text': 'output text'}]}] |

## Troubleshooting

### OSError: [Errno 24] Too many open files

Reference: https://stackoverflow.com/a/39537952

### Killed (when run train script get this error)

Not enough memory. need increase memory or swap space.

Increase swap space can reference: https://askubuntu.com/questions/178712/how-to-increase-swap-space/534090#534090
