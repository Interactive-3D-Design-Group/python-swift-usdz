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
    # ``\operatorname{abs}`` normalizes to ``operatorname{abs}``; brace extraction must not treat ``{abs}``
    # as a domain group — collapse to plain ``abs(...)`` first.
    text = text.replace("operatorname{abs}", "abs")
    # ``abs(x)<c`` inside ``{...}`` domain — rewrite to a chained compare so ``extract_brace_restrictions`` works.
    def _abs_symmetric_bounds(m: re.Match[str]) -> str:
        axis, op, bound = m.group(1), m.group(2), m.group(3).strip()
        if op == "<=":
            return f"-({bound})<={axis}<={bound}"
        return f"-({bound})<{axis}<{bound}"

    # Bound must not swallow a following ``{...}`` domain brace — exclude ``{`` (and ``}``) from the capture.
    text = re.sub(
        r"abs\(([xyz])\)(<=|<)([^,{}]+)",
        _abs_symmetric_bounds,
        text,
    )
    # Strip exponent/subscript braces so nested ``sqrt{...^{2}...}`` can be parsed later.
    text = re.sub(r"\^\{([^{}]+)\}", r"^\1", text)
    text = re.sub(r"_\{([^{}]+)\}", r"_\1", text)
    # ``2piu`` -> ``2*pi*u`` (digit before ``pi`` only — do not turn ``cos(piu`` into ``cos(*piu``).
    text = re.sub(r"([0-9])pi(?=[uvt])", r"\1*pi*", text)
    for trig in ("cos", "sin", "tan"):
        text = re.sub(rf"(?<={trig}\()pi(?=[uvt])", "pi*", text)
    # Desmos slider/list shorthand ``[a,b,...c]`` inside arithmetic — keep first value for parsing.
    text = re.sub(r"\[([0-9]+(?:\.[0-9]*)?)[\s,][^\]]*\.\.[^\]]+\]", r"(\1)", text)
    # Two labeled points in brackets → segment (common diagram shorthand).
    text = re.sub(
        r"\[([A-Za-z][A-Za-z0-9_]*),([A-Za-z][A-Za-z0-9_]*)\]",
        r"operatorname{segment}((\1),(\2))",
        text,
    )
    # Double-wrapped numeric / simple point triple: ``G=((20,20,0))`` → ``G=(20,20,0)``.
    for _ in range(4):
        ntext = re.sub(
            r"=+\(\(([^,()]{1,120}),([^,()]{1,120}),([^)()]{1,120})\)\)",
            r"=(\1,\2,\3)",
            text,
        )
        if ntext == text:
            break
        text = ntext
    return text


def _flatten_embedded_intervals(inner: str) -> list[str]:
    """Split ``0<z<23{-2<y<2}``-style embedded groups into separate interval strings."""
    inner = inner.strip()
    if "{" not in inner:
        return [inner] if inner else []
    j = inner.index("{")
    depth = 0
    for k in range(j, len(inner)):
        if inner[k] == "{":
            depth += 1
        elif inner[k] == "}":
            depth -= 1
            if depth == 0:
                before = inner[:j].strip()
                mid = inner[j + 1 : k].strip()
                after = inner[k + 1 :].strip()
                out: list[str] = []
                if before:
                    out.append(before)
                out.extend(_flatten_embedded_intervals(mid))
                if after:
                    out.extend(_flatten_embedded_intervals(after))
                return out
    return [inner]


def _top_level_brace_spans(s: str) -> list[tuple[int, int, str]]:
    """Each entry is ``(start_index, end_index_exclusive, inner)`` for a matched ``{...}`` at depth 0."""
    spans: list[tuple[int, int, str]] = []
    depth = 0
    start: int | None = None
    inner_start: int | None = None
    for i, ch in enumerate(s):
        if ch == "{":
            if depth == 0:
                start = i
                inner_start = i + 1
            depth += 1
        elif ch == "}":
            if depth == 0:
                continue
            depth -= 1
            if depth == 0 and start is not None and inner_start is not None:
                inner = s[inner_start:i]
                spans.append((start, i + 1, inner))
                start = None
                inner_start = None
    return spans


def extract_brace_restrictions(normalized: str) -> tuple[str, list[str]]:
    text = re.sub(r"-\{([0-9]+(?:\.[0-9]+)?<=)", r"{-\1", normalized)
    # Normalize subscript/exponent braces so restriction parsing can handle them inside {...}.
    # Example: {-14.5<=z<=0.0007(x+505)^{2}-16} contains inner {2} which would break a simple {[^{}]+} matcher.
    text = re.sub(r"\^\{([^{}]+)\}", r"^\1", text)
    text = re.sub(r"_\{([^{}]+)\}", r"_\1", text)
    # ``sqrt{2}`` must not be parsed as a top-level ``{2}`` domain brace.
    text = re.sub(r"sqrt\{([^{}]+)\}", r"sqrt(\1)", text)
    spans = _top_level_brace_spans(text)
    restrictions: list[str] = []
    core_parts: list[str] = []
    last = 0
    for s, e, inner in spans:
        core_parts.append(text[last:s])
        restrictions.extend(_flatten_embedded_intervals(inner))
        last = e
    core_parts.append(text[last:])
    core = "".join(core_parts).rstrip("-")
    while "()" in core:
        core = core.replace("()", "")
    return core, restrictions
