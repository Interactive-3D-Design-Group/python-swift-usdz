from __future__ import annotations

from desmos3d_pipeline.classify.rules import classify_expression
from desmos3d_pipeline.ir.models import ClassificationStatus, ExpressionFamily
from desmos3d_pipeline.normalize.latex import extract_brace_restrictions, normalize_latex
from desmos3d_pipeline.parse.relation import parse_interval_constraint


def test_normalize_latex_desmos_tokens() -> None:
    raw = r"z\le\frac{x}{2}\left\{-3<x<3\right\}"
    normalized = normalize_latex(raw)
    assert "<=" in normalized
    assert "(x)/(2)" in normalized
    assert "{" in normalized and "}" in normalized


def test_extract_brace_restrictions() -> None:
    core, restrictions = extract_brace_restrictions("z=7{-3<x<3}{-26<y<26}")
    assert core == "z=7"
    assert restrictions == ["-3<x<3", "-26<y<26"]


def test_parse_interval_constraint_chained() -> None:
    constraint = parse_interval_constraint("-5<=x<=5")
    assert constraint is not None
    assert constraint.axis == "x"
    assert constraint.lower == "-5"
    assert constraint.upper == "5"


def test_classify_constant_plane_supported() -> None:
    result = classify_expression("z=80{100>x>60}{20>y>-20}", "expression")
    assert result.family == ExpressionFamily.CONSTANT_PLANE
    assert result.status == ClassificationStatus.SUPPORTED


def test_classify_box_supported() -> None:
    result = classify_expression("-5<=x<=5{-80<=z<=171}{20<=y<=25}", "expression")
    assert result.family == ExpressionFamily.BOX_BOUNDED_REGION
    assert result.status == ClassificationStatus.SUPPORTED


def test_classify_unknown() -> None:
    result = classify_expression("thisIsNotValid", "expression")
    assert result.status == ClassificationStatus.UNRECOGNIZED
