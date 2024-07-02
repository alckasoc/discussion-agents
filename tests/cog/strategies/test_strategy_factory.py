"""Unit tests for strategy factory classes."""

from unittest.mock import MagicMock

import pytest

from langchain_community.chat_models.fake import FakeListChatModel
from langchain_community.utilities.google_serper import GoogleSerperAPIWrapper

from agential.cog.strategies.critic.code import (
    CritHEvalCodeStrategy,
    CritMBPPCodeStrategy,
)
from agential.cog.strategies.critic.math import (
    CritGSM8KStrategy,
    CritSVAMPStrategy,
    CritTabMWPStrategy,
)
from agential.cog.strategies.critic.qa import (
    CritAmbigNQStrategy,
    CritFEVERStrategy,
    CritHotQAStrategy,
    CritTriviaQAStrategy,
)
from agential.cog.strategies.react.code import ReActHEvalStrategy, ReActMBPPStrategy
from agential.cog.strategies.react.math import (
    ReActGSM8KStrategy,
    ReActSVAMPStrategy,
    ReActTabMWPStrategy,
)
from agential.cog.strategies.react.qa import (
    ReActAmbigNQStrategy,
    ReActFEVERStrategy,
    ReActHotQAStrategy,
    ReActTriviaQAStrategy,
)
from agential.cog.strategies.reflexion.code import (
    ReflexionCoTHEvalStrategy,
    ReflexionCoTMBPPStrategy,
    ReflexionReActHEvalStrategy,
    ReflexionReActMBPPStrategy,
)
from agential.cog.strategies.reflexion.math import (
    ReflexionCoTGSM8KStrategy,
    ReflexionCoTSVAMPStrategy,
    ReflexionCoTTabMWPStrategy,
    ReflexionReActGSM8KStrategy,
    ReflexionReActSVAMPStrategy,
    ReflexionReActTabMWPStrategy,
)
from agential.cog.strategies.reflexion.qa import (
    ReflexionCoTAmbigNQStrategy,
    ReflexionCoTFEVERStrategy,
    ReflexionCoTHotQAStrategy,
    ReflexionCoTTriviaQAStrategy,
    ReflexionReActAmbigNQStrategy,
    ReflexionReActFEVERStrategy,
    ReflexionReActHotQAStrategy,
    ReflexionReActTriviaQAStrategy,
)
from agential.cog.strategies.strategy_factory import (
    CriticStrategyFactory,
    ReActStrategyFactory,
    ReflexionCoTStrategyFactory,
    ReflexionReActStrategyFactory,
)


def test_critic_strategy_factory_get_strategy() -> None:
    """Tests CriticStrategyFactory get_strategy method."""
    search = MagicMock(spec=GoogleSerperAPIWrapper)

    llm = FakeListChatModel(responses=[])

    # QA benchmarks.
    assert isinstance(
        CriticStrategyFactory.get_strategy({"qa": "hotpotqa"}, llm=llm),
        CritHotQAStrategy,
    )
    assert isinstance(
        CriticStrategyFactory.get_strategy({"qa": "triviaqa"}, llm=llm),
        CritTriviaQAStrategy,
    )
    assert isinstance(
        CriticStrategyFactory.get_strategy({"qa": "ambignq"}, llm=llm),
        CritAmbigNQStrategy,
    )
    assert isinstance(
        CriticStrategyFactory.get_strategy({"qa": "fever"}, llm=llm), CritFEVERStrategy
    )

    # Test kwargs for QA strategy.
    strategy = CriticStrategyFactory.get_strategy(
        {"qa": "fever"}, llm=llm, search=search, evidence_length=500, num_results=10
    )
    assert isinstance(strategy, CritFEVERStrategy)
    assert strategy.llm == llm
    assert isinstance(strategy.search, GoogleSerperAPIWrapper)
    assert strategy.evidence_length == 500
    assert strategy.num_results == 10

    # Math benchmarks.
    assert isinstance(
        CriticStrategyFactory.get_strategy({"math": "gsm8k"}, llm=llm),
        CritGSM8KStrategy,
    )
    assert isinstance(
        CriticStrategyFactory.get_strategy({"math": "svamp"}, llm=llm),
        CritSVAMPStrategy,
    )
    assert isinstance(
        CriticStrategyFactory.get_strategy({"math": "tabmwp"}, llm=llm),
        CritTabMWPStrategy,
    )

    # Test kwargs for Math strategy.
    strategy = CriticStrategyFactory.get_strategy(
        {"math": "gsm8k"}, llm=llm, patience=3
    )
    assert isinstance(strategy, CritGSM8KStrategy)
    assert strategy.llm == llm
    assert strategy.patience == 3

    # Code benchmarks.
    assert isinstance(
        CriticStrategyFactory.get_strategy({"code": "mbpp"}, llm=llm),
        CritMBPPCodeStrategy,
    )
    assert isinstance(
        CriticStrategyFactory.get_strategy({"code": "humaneval"}, llm=llm),
        CritHEvalCodeStrategy,
    )

    # Unsupported benchmarks.
    with pytest.raises(ValueError, match="Unsupported QA benchmark: unknown"):
        CriticStrategyFactory.get_strategy({"qa": "unknown"})

    with pytest.raises(ValueError, match="Unsupported Math benchmark: unknown"):
        CriticStrategyFactory.get_strategy({"math": "unknown"})

    with pytest.raises(ValueError, match="Unsupported Code benchmark: unknown"):
        CriticStrategyFactory.get_strategy({"code": "unknown"})

    with pytest.raises(ValueError, match="Unsupported mode: {}"):
        CriticStrategyFactory.get_strategy({})


def test_react_strategy_factory_get_strategy() -> None:
    """Tests ReActStrategyFactory get_strategy method."""
    llm = FakeListChatModel(responses=[])

    # QA benchmarks.
    assert isinstance(
        ReActStrategyFactory.get_strategy({"qa": "hotpotqa"}, llm=llm),
        ReActHotQAStrategy,
    )
    assert isinstance(
        ReActStrategyFactory.get_strategy({"qa": "triviaqa"}, llm=llm),
        ReActTriviaQAStrategy,
    )
    assert isinstance(
        ReActStrategyFactory.get_strategy({"qa": "ambignq"}, llm=llm),
        ReActAmbigNQStrategy,
    )
    assert isinstance(
        ReActStrategyFactory.get_strategy({"qa": "fever"}, llm=llm),
        ReActFEVERStrategy,
    )

    # Test kwargs for QA strategy.
    strategy = ReActStrategyFactory.get_strategy({"qa": "hotpotqa"}, llm=llm)
    assert isinstance(strategy, ReActHotQAStrategy)
    assert strategy.llm == llm

    # Math benchmarks.
    assert isinstance(
        ReActStrategyFactory.get_strategy({"math": "gsm8k"}, llm=llm),
        ReActGSM8KStrategy,
    )
    assert isinstance(
        ReActStrategyFactory.get_strategy({"math": "svamp"}, llm=llm),
        ReActSVAMPStrategy,
    )
    assert isinstance(
        ReActStrategyFactory.get_strategy({"math": "tabmwp"}, llm=llm),
        ReActTabMWPStrategy,
    )

    # Test kwargs for Math strategy.
    strategy = ReActStrategyFactory.get_strategy(
        {"math": "gsm8k"}, llm=llm, max_tokens=123
    )
    assert isinstance(strategy, ReActGSM8KStrategy)
    assert strategy.llm == llm
    assert strategy.max_tokens == 123

    # Code benchmarks.
    assert isinstance(
        ReActStrategyFactory.get_strategy({"code": "humaneval"}, llm=llm),
        ReActHEvalStrategy,
    )

    assert isinstance(
        ReActStrategyFactory.get_strategy({"code": "mbpp"}, llm=llm),
        ReActMBPPStrategy,
    )

    # Test kwargs for Code strategy.
    strategy = ReActStrategyFactory.get_strategy(
        {"code": "mbpp"}, llm=llm, max_tokens=123
    )
    assert isinstance(strategy, ReActMBPPStrategy)
    assert strategy.llm == llm
    assert strategy.max_tokens == 123

    # Unsupported benchmarks.
    with pytest.raises(ValueError, match="Unsupported QA benchmark: unknown"):
        ReActStrategyFactory.get_strategy({"qa": "unknown"})

    with pytest.raises(ValueError, match="Unsupported Math benchmark: unknown"):
        ReActStrategyFactory.get_strategy({"math": "unknown"})

    with pytest.raises(ValueError, match="Unsupported Code benchmark: unknown"):
        ReActStrategyFactory.get_strategy({"code": "unknown"})

    with pytest.raises(ValueError, match="Unsupported mode: {}"):
        ReActStrategyFactory.get_strategy({})


def test_reflexioncot_strategy_factory_get_strategy() -> None:
    """Tests ReflexionCoTStrategyFactory get_strategy method."""
    llm = FakeListChatModel(responses=[])

    # QA benchmarks.
    assert isinstance(
        ReflexionCoTStrategyFactory.get_strategy({"qa": "hotpotqa"}, llm=llm),
        ReflexionCoTHotQAStrategy,
    )
    assert isinstance(
        ReflexionCoTStrategyFactory.get_strategy({"qa": "triviaqa"}, llm=llm),
        ReflexionCoTTriviaQAStrategy,
    )
    assert isinstance(
        ReflexionCoTStrategyFactory.get_strategy({"qa": "ambignq"}, llm=llm),
        ReflexionCoTAmbigNQStrategy,
    )
    assert isinstance(
        ReflexionCoTStrategyFactory.get_strategy({"qa": "fever"}, llm=llm),
        ReflexionCoTFEVERStrategy,
    )

    # Math benchmarks.
    assert isinstance(
        ReflexionCoTStrategyFactory.get_strategy({"math": "gsm8k"}, llm=llm),
        ReflexionCoTGSM8KStrategy,
    )
    assert isinstance(
        ReflexionCoTStrategyFactory.get_strategy({"math": "svamp"}, llm=llm),
        ReflexionCoTSVAMPStrategy,
    )
    assert isinstance(
        ReflexionCoTStrategyFactory.get_strategy({"math": "tabmwp"}, llm=llm),
        ReflexionCoTTabMWPStrategy,
    )

    # Code benchmarks.
    assert isinstance(
        ReflexionCoTStrategyFactory.get_strategy({"code": "humaneval"}, llm=llm),
        ReflexionCoTHEvalStrategy,
    )
    assert isinstance(
        ReflexionCoTStrategyFactory.get_strategy({"code": "mbpp"}, llm=llm),
        ReflexionCoTMBPPStrategy,
    )

    # Test kwargs for QA strategy.
    strategy = ReflexionCoTStrategyFactory.get_strategy(
        {"qa": "hotpotqa"}, llm=llm, max_reflections=1
    )
    assert isinstance(strategy, ReflexionCoTHotQAStrategy)
    assert strategy.llm == llm
    assert strategy.max_reflections == 1

    # Unsupported benchmarks.
    with pytest.raises(ValueError, match="Unsupported QA benchmark: unknown"):
        ReflexionCoTStrategyFactory.get_strategy({"qa": "unknown"})

    with pytest.raises(ValueError, match="Unsupported Math benchmark: unknown"):
        ReflexionCoTStrategyFactory.get_strategy({"math": "unknown"})

    with pytest.raises(ValueError, match="Unsupported Code benchmark: unknown"):
        ReflexionCoTStrategyFactory.get_strategy({"code": "unknown"})

    with pytest.raises(ValueError, match="Unsupported mode: {}"):
        ReflexionCoTStrategyFactory.get_strategy({})


def test_reflexionreact_strategy_factory_get_strategy() -> None:
    """Tests ReflexionReActStrategyFactory get_strategy method."""
    llm = FakeListChatModel(responses=[])

    # QA benchmarks.
    assert isinstance(
        ReflexionReActStrategyFactory.get_strategy({"qa": "hotpotqa"}, llm=llm),
        ReflexionReActHotQAStrategy,
    )
    assert isinstance(
        ReflexionReActStrategyFactory.get_strategy({"qa": "triviaqa"}, llm=llm),
        ReflexionReActTriviaQAStrategy,
    )
    assert isinstance(
        ReflexionReActStrategyFactory.get_strategy({"qa": "ambignq"}, llm=llm),
        ReflexionReActAmbigNQStrategy,
    )
    assert isinstance(
        ReflexionReActStrategyFactory.get_strategy({"qa": "fever"}, llm=llm),
        ReflexionReActFEVERStrategy,
    )

    # Math benchmarks.
    assert isinstance(
        ReflexionReActStrategyFactory.get_strategy({"math": "gsm8k"}, llm=llm),
        ReflexionReActGSM8KStrategy,
    )
    assert isinstance(
        ReflexionReActStrategyFactory.get_strategy({"math": "svamp"}, llm=llm),
        ReflexionReActSVAMPStrategy,
    )
    assert isinstance(
        ReflexionReActStrategyFactory.get_strategy({"math": "tabmwp"}, llm=llm),
        ReflexionReActTabMWPStrategy,
    )

    # Code benchmarks.
    assert isinstance(
        ReflexionReActStrategyFactory.get_strategy({"code": "humaneval"}, llm=llm),
        ReflexionReActHEvalStrategy,
    )
    assert isinstance(
        ReflexionReActStrategyFactory.get_strategy({"code": "mbpp"}, llm=llm),
        ReflexionReActMBPPStrategy,
    )

    # Test kwargs for QA strategy.
    strategy = ReflexionReActStrategyFactory.get_strategy(
        {"qa": "hotpotqa"}, llm=llm, max_reflections=1
    )
    assert isinstance(strategy, ReflexionReActHotQAStrategy)
    assert strategy.llm == llm
    assert strategy.max_reflections == 1

    # Unsupported benchmarks.
    with pytest.raises(ValueError, match="Unsupported QA benchmark: unknown"):
        ReflexionReActStrategyFactory.get_strategy({"qa": "unknown"})

    with pytest.raises(ValueError, match="Unsupported Math benchmark: unknown"):
        ReflexionReActStrategyFactory.get_strategy({"math": "unknown"})

    with pytest.raises(ValueError, match="Unsupported Code benchmark: unknown"):
        ReflexionReActStrategyFactory.get_strategy({"code": "unknown"})

    with pytest.raises(ValueError, match="Unsupported mode: {}"):
        ReflexionReActStrategyFactory.get_strategy({})
