# coding: utf-8

from __future__ import print_function

import string
from pathlib import Path

from unsloth.tokenizer_utils import logger


class SafeFormatter(string.Formatter):
    """Reference: https://stackoverflow.com/a/34033230"""

    def vformat(self, format_string, args, kwargs):
        args_len = len(args)  # for checking IndexError
        tokens = []
        for lit, name, spec, conv in self.parse(format_string):
            # re-escape braces that parse() unescaped
            lit = lit.replace("{", "{{").replace("}", "}}")
            # only lit is non-None at the end of the string
            if name is None:
                tokens.append(lit)
            else:
                # but conv and spec are None if unused
                conv = "!" + conv if conv else ""
                spec = ":" + spec if spec else ""
                # name includes indexing ([blah]) and attributes (.blah)
                # so get just the first part
                fp = name.split("[")[0].split(".")[0]
                # treat as normal if fp is empty (an implicit
                # positional arg), a digit (an explicit positional
                # arg) or if it is in kwargs
                if not fp or fp.isdigit() or fp in kwargs:
                    tokens.extend([lit, "{", name, conv, spec, "}"])
                # otherwise escape the braces
                else:
                    tokens.extend([lit, "{{", name, conv, spec, "}}"])
        format_string = "".join(tokens)  # put the string back together
        # finally call the default formatter
        return string.Formatter.vformat(self, format_string, args, kwargs)


def create_ollama_modelfile(tokenizer, gguf_location):
    """
    Creates an Ollama Modelfile.
    Use ollama.create(model = "new_ollama_model", modelfile = modelfile)
    """
    modelfile = getattr(tokenizer, "_ollama_modelfile", None)
    if modelfile is None:
        return None

    modelfile = modelfile.replace("{{", "⚫@✅#🦥").replace("}}", "⚡@🦥#⛵")

    model_name = Path(gguf_location).name
    if "__EOS_TOKEN__" in modelfile:
        modelfile = SafeFormatter().format(
            modelfile,
            __FILE_LOCATION__=f"./{model_name}",
            __EOS_TOKEN__=tokenizer.eos_token,
        )
    else:
        modelfile = SafeFormatter().format(
            modelfile,
            __FILE_LOCATION__=f"./{model_name}",
        )

    modelfile = modelfile.replace("⚫@✅#🦥", "{{").replace("⚡@🦥#⛵", "}}").rstrip()

    return modelfile


def _fix_chat_template(chat_template):
    endfor = "{% endfor %}"
    where = chat_template.find(endfor)
    if where == -1:
        return chat_template

    after_endfor = chat_template[where + len(endfor) :]

    if (
        "{% if" not in after_endfor
        and "{% set " not in after_endfor
        and after_endfor.startswith("{{")
        and after_endfor.endswith("}}")
        and after_endfor.count("{{") == 1
        and after_endfor.count("}}") == 1
    ):
        after_endfor = "{% if add_generation_prompt %}" + after_endfor + "{% endif %}"

        chat_template = chat_template[: where + len(endfor)] + after_endfor
    pass
    return chat_template


pass


def fix_chat_template(tokenizer):
    chat_template = getattr(tokenizer, "chat_template", None)
    if chat_template is None:
        return None

    ### 1. Check if add_generation_prompt works
    # Check for ShareGPT style first
    is_sharegpt = None
    try:
        messages = [
            {"role": "user", "content": "Who are you?"},
        ]
        tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
        is_sharegpt = False
    except Exception:
        try:
            messages = [
                {"from": "human", "value": "Who are you?"},
            ]
            tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
            is_sharegpt = True
        except Exception:
            is_sharegpt = None
        pass
    pass

    # Not ShareGPT or HF style - just return
    if is_sharegpt is None:
        return chat_template

    # Tokenize
    messages = [{"role": "user", "content": "Who are you?"} if not is_sharegpt else {"from": "human", "value": "Who are you?"}]
    no = tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
    yes = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    if no == yes:
        # SAME?! That's not good! We check for add_generation_prompt
        if "{% if add_generation_prompt %}" not in chat_template:
            # Try fixing it by adding it
            new_chat_template = _fix_chat_template(chat_template)
            if "{% if add_generation_prompt %}" not in new_chat_template:
                raise RuntimeError(
                    f"Unsloth: The tokenizer `{tokenizer.name_or_path}`\n"
                    "does not have a {% if add_generation_prompt %} for generation purposes.\n"
                    "Please file a bug report immediately - thanks!"
                )
            else:
                logger.warning_once(
                    "Unsloth: We successfully patched the tokenizer to add a {% if add_generation_prompt %} to the chat_template.\n"
                    "This is not a bug, but please notify the Unsloth maintainers - thanks!"
                )
                chat_template = new_chat_template
            pass
        else:
            raise RuntimeError(
                f"Unsloth: The tokenizer `{tokenizer.name_or_path}`\n"
                "has a {% if add_generation_prompt %} for generation purposes, but wasn't provided correctly.\n"
                "Please file a bug report immediately - thanks!"
            )
        pass
    pass
    return chat_template


pass
