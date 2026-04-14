from __future__ import annotations

import re

REPLACEMENTS = {
    r"\left": "",
    r"\right": "",
    r"\le": "<=",
    r"\ge": ">=",
    r"\cdot": "*",
    r"\{": "{",
    r"\}": "}",
    r"\ ": " ",
    "−": "-",
}


def normalize_latex(raw: str) -> str:
    text = raw
    for src, dst in REPLACEMENTS.items():
        text = text.replace(src, dst)
    text = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", text)
    text = text.replace("\\", "")
    text = re.sub(r"\s+", "", text)
    text = text.replace("+-", "-")
    text = text.replace("--", "+")
    return text


def extract_brace_restrictions(normalized: str) -> tuple[str, list[str]]:
    text = re.sub(r"-\{([0-9]+(?:\.[0-9]+)?<=)", r"{-\1", normalized)
    # Normalize subscript/exponent braces so restriction parsing can handle them inside {...}.
    # Example: {-14.5<=z<=0.0007(x+505)^{2}-16} contains inner {2} which would break a simple {[^{}]+} matcher.
    text = re.sub(r"\^\{([^{}]+)\}", r"^\1", text)
    text = re.sub(r"_\{([^{}]+)\}", r"_\1", text)
    # Do not treat subscript/exponent braces (e.g. a_{5}, x^{2}) as domain restrictions.
    restrictions = re.findall(r"(?<![_^])\{([^{}]+)\}", text)
    core = re.sub(r"(?<![_^])\{[^{}]+\}", "", text)
    core = core.rstrip("-")
    return core, restrictions
