# unsloth-train

A use unsloth to train llm model script

## Install

```bash
# If create venv
# python -m venv ./.venv
# . ./.venv/bin/activate

python -m pip install --upgrade pip wheel setuptools
pip install -r requirements.txt
```

## Train model

```bash
python unsloth_train/train.py
```

## Change setting and data path

At `unsloth_train/train.py` line 149-161
