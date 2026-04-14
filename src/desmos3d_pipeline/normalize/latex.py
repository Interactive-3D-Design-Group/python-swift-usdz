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
    restrictions = re.findall(r"\{([^{}]+)\}", text)
    core = re.sub(r"\{[^{}]+\}", "", text)
    core = core.rstrip("-")
    return core, restrictions
