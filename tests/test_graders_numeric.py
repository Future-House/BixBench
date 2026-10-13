from unittest.mock import AsyncMock

import pytest

from bixbench.graders import GradingFunction, MCQGrader


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("target", "predicted", "expected"),
    [
        ("-4.96", "4.96", False),
        ("0.0002", "00002", False),
        ("1.9E-05", "1.9E05", False),
        ("4.25E-04", "5", False),
        ("1.0", "1e0", True),
        (".5", "0.5", True),
        ("9007199254740993", "9007199254740992", False),
        ("1", "NaN", False),
        ("1", "Infinity", False),
        ("12", "1 2", False),
        ("-1 mg", "1 mg", False),
        ("5 mg", "5 g", False),
        ("5%", "5", False),
        ("H2O", "H2", False),
        ("dose15mg", "dose1mg", False),
        ("IL10", "IL1", False),
        ("New York", "York", True),
        ("hello, world!", "HELLO WORLD", True),
        ("answer", "", False),
        ("answer", "!!!", False),
    ],
)
async def test_numeric_boundaries(target, predicted, expected):
    r = await GradingFunction()._grade_str_verifier(
        target, predicted, partial_match=True, llm_match=False
    )
    assert r.correct == expected


@pytest.mark.asyncio
async def test_unequal_scalars_do_not_call_semantic_verifier():
    client = AsyncMock()
    r = await GradingFunction()._grade_str_verifier(
        "-1", "1", partial_match=True, llm_match=True, llm_client=client
    )
    assert not r.correct
    client.call_single.assert_not_called()


@pytest.mark.asyncio
async def test_mcq_numeric_labels_are_not_numeric_equivalence():
    assert not (await MCQGrader().grade("1", "1.0")).correct
    assert (await MCQGrader().grade("A", "(a)")).correct


@pytest.mark.asyncio
async def test_mixed_units_use_only_explicit_semantic_fallback():
    client = AsyncMock()
    client.call_single.return_value.text = "<grade>incorrect</grade>"
    r = await GradingFunction()._grade_str_verifier(
        "5 mg", "5 g", partial_match=True, llm_match=True, llm_client=client
    )
    assert not r.correct
    client.call_single.assert_awaited_once()
