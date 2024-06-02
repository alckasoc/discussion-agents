"""Unit tests for ReAct math strategies."""

from langchain_community.chat_models.fake import FakeListChatModel
from langchain_core.language_models.chat_models import BaseChatModel
from tiktoken import Encoding

from agential.cog.strategies.react.math import (
    ReActMathStrategy,
    ReActGSM8KStrategy,
    ReActSVAMPStrategy,
    ReActTabMWPStrategy,
    parse_math_action,
)


def test_parse_math_action() -> None:
    """Test parse_math_action."""
    test_cases = [
        {
            "input": "Calculate[```python\ndef add(a, b): return a + b\n```]",
            "expected": ("Calculate", "def add(a, b): return a + b"),
        },
        {
            "input": "Finish[```python\nassert add(2, 3) == 5\n```]",
            "expected": ("Finish", "assert add(2, 3) == 5"),
        },
        {
            "input": "Finish[```python\nThe function is complete.\n```]",
            "expected": ("Finish", "The function is complete."),
        },
        {
            "input": "calculate[```python\ndef subtract(a, b): return a - b\n```]",
            "expected": ("Calculate", "def subtract(a, b): return a - b"),
        },
        {
            "input": "Invalid[```python\nThis should not match\n```]",
            "expected": ("", ""),
        },
        {
            "input": "Calculate[```python\nassert subtract(5, 3) == 2\n```]",
            "expected": ("Calculate", "assert subtract(5, 3) == 2"),
        },
        {
            "input": "Something else entirely",
            "expected": ("", ""),
        },
        {
            "input": "Finish[```python\n \n```]",
            "expected": ("Finish", ""),
        },
        {
            "input": "Calculate[```python\nfor i in range(10):\n    print(i)\n```]",
            "expected": ("Calculate", "for i in range(10):\n    print(i)"),
        },
    ]

    for case in test_cases:
        result = parse_math_action(case["input"])
        assert result == case["expected"]


def test_init() -> None:
    """Test ReActMathStrategy initialization."""
    llm = FakeListChatModel(responses=[])
    strategy = ReActMathStrategy(llm=llm)
    assert isinstance(strategy.llm, BaseChatModel)
    assert strategy.max_steps == 6
    assert strategy.max_tokens == 3896
    assert isinstance(strategy.enc, Encoding)
    assert strategy._current_answer == ""
    assert strategy._scratchpad == ""
    assert strategy._finished == False