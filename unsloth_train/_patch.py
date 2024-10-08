# coding: utf-8

from __future__ import print_function

import string


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

    if "__EOS_TOKEN__" in modelfile:
        modelfile = SafeFormatter().format(
            modelfile,
            __FILE_LOCATION__=gguf_location,
            __EOS_TOKEN__=tokenizer.eos_token,
        )
    else:
        modelfile = SafeFormatter().format(
            modelfile,
            __FILE_LOCATION__=gguf_location,
        )

    modelfile = modelfile.replace("⚫@✅#🦥", "{{").replace("⚡@🦥#⛵", "}}").rstrip()

    return modelfile
