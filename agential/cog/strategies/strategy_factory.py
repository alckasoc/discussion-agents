"""Strategy factory classes."""

from typing import Any, Dict

from agential.cog.strategies.critic.base import CriticBaseStrategy
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
from agential.cog.strategies.react.base import ReActBaseStrategy
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
from agential.cog.strategies.reflexion.base import (
    ReflexionCoTBaseStrategy,
    ReflexionReActBaseStrategy,
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


class CriticStrategyFactory:
    """A factory class for creating instances of different CRITIC strategies based on the specified mode and benchmark.

    Methods:
        get_strategy(mode: Dict[str, str], **strategy_kwargs) -> CriticBaseStrategy:
            Returns an instance of the appropriate Critic strategy based on the provided mode and benchmark.
    """

    @staticmethod
    def get_strategy(
        mode: Dict[str, str], **strategy_kwargs: Any
    ) -> CriticBaseStrategy:
        """Returns an instance of the appropriate Critic strategy based on the provided mode and benchmark.

        Available modes:
            - qa: "hotpotqa", "triviaqa", "ambignq", "fever"
            - math: "gsm8k", "svamp", "tabmwp"
            - code: "mbpp", "humaneval"

        Args:
            mode (Dict[str, str]): A dictionary specifying the mode and benchmark.
                Example: {"qa": "hotpotqa"}, {"math": "gsm8k"}, {"code": "mbpp"}.
            **strategy_kwargs (Dict[str, Any]): Additional keyword arguments to pass to the strategy's constructor.

        Returns:
            CriticBaseStrategy: An instance of the appropriate Critic strategy.

        Raises:
            ValueError: If the mode or benchmark is unsupported.
        """
        if "qa" in mode:
            if mode["qa"] == "hotpotqa":
                return CritHotQAStrategy(**strategy_kwargs)
            elif mode["qa"] == "triviaqa":
                return CritTriviaQAStrategy(**strategy_kwargs)
            elif mode["qa"] == "ambignq":
                return CritAmbigNQStrategy(**strategy_kwargs)
            elif mode["qa"] == "fever":
                return CritFEVERStrategy(**strategy_kwargs)
            else:
                raise ValueError(f"Unsupported QA benchmark: {mode['qa']}")
        elif "math" in mode:
            if mode["math"] == "gsm8k":
                return CritGSM8KStrategy(**strategy_kwargs)
            elif mode["math"] == "svamp":
                return CritSVAMPStrategy(**strategy_kwargs)
            elif mode["math"] == "tabmwp":
                return CritTabMWPStrategy(**strategy_kwargs)
            else:
                raise ValueError(f"Unsupported Math benchmark: {mode['math']}")
        elif "code" in mode:
            if mode["code"] == "humaneval":
                return CritHEvalCodeStrategy(**strategy_kwargs)
            elif mode["code"] == "mbpp":
                return CritMBPPCodeStrategy(**strategy_kwargs)
            else:
                raise ValueError(f"Unsupported Code benchmark: {mode['code']}")
        else:
            raise ValueError(f"Unsupported mode: {mode}")


class ReActStrategyFactory:
    """A factory class for creating instances of different ReAct strategies based on the specified mode and benchmark.

    Methods:
        get_strategy(mode: Dict[str, str], **strategy_kwargs) -> ReActBaseStrategy:
            Returns an instance of the appropriate ReAct strategy based on the provided mode and benchmark.
    """

    @staticmethod
    def get_strategy(mode: Dict[str, str], **strategy_kwargs: Any) -> ReActBaseStrategy:
        """Returns an instance of the appropriate ReAct strategy based on the provided mode and benchmark.

        Available modes:
            - qa: "hotpotqa", "triviaqa", "ambignq", "fever"
            - math: "gsm8k", "svamp", "tabmwp"
            - code: "mbpp", "humaneval"

        Args:
            mode (Dict[str, str]): A dictionary specifying the mode and benchmark.
                Example: {"qa": "hotpotqa"}, {"math": "gsm8k"}, {"code": "mbpp"}.
            **strategy_kwargs (Dict[str, Any]): Additional keyword arguments to pass to the strategy's constructor.

        Returns:
            ReActBaseStrategy: An instance of the appropriate ReAct strategy.

        Raises:
            ValueError: If the mode or benchmark is unsupported.
        """
        if "qa" in mode:
            if mode["qa"] == "hotpotqa":
                return ReActHotQAStrategy(**strategy_kwargs)
            elif mode["qa"] == "triviaqa":
                return ReActTriviaQAStrategy(**strategy_kwargs)
            elif mode["qa"] == "ambignq":
                return ReActAmbigNQStrategy(**strategy_kwargs)
            elif mode["qa"] == "fever":
                return ReActFEVERStrategy(**strategy_kwargs)
            else:
                raise ValueError(f"Unsupported QA benchmark: {mode['qa']}")
        elif "math" in mode:
            if mode["math"] == "gsm8k":
                return ReActGSM8KStrategy(**strategy_kwargs)
            elif mode["math"] == "svamp":
                return ReActSVAMPStrategy(**strategy_kwargs)
            elif mode["math"] == "tabmwp":
                return ReActTabMWPStrategy(**strategy_kwargs)
            else:
                raise ValueError(f"Unsupported Math benchmark: {mode['math']}")
        elif "code" in mode:
            if mode["code"] == "humaneval":
                return ReActHEvalStrategy(**strategy_kwargs)
            elif mode["code"] == "mbpp":
                return ReActMBPPStrategy(**strategy_kwargs)
            else:
                raise ValueError(f"Unsupported Code benchmark: {mode['code']}")
        else:
            raise ValueError(f"Unsupported mode: {mode}")


class ReflexionCoTStrategyFactory:
    """A factory class for creating instances of different ReflexionCoT strategies based on the specified mode and benchmark.

    Methods:
        get_strategy(mode: Dict[str, str], **strategy_kwargs) -> ReflexionCoTBaseStrategy:
            Returns an instance of the appropriate ReflexionCoT strategy based on the provided mode and benchmark.
    """

    @staticmethod
    def get_strategy(
        mode: Dict[str, str], **strategy_kwargs: Any
    ) -> ReflexionCoTBaseStrategy:
        """Returns an instance of the appropriate ReflexionCoT strategy based on the provided mode and benchmark.

        Available modes:
            - qa: "hotpotqa", "triviaqa", "ambignq", "fever"
            - math: "gsm8k", "svamp", "tabmwp"
            - code: "mbpp", "humaneval"

        Args:
            mode (Dict[str, str]): A dictionary specifying the mode and benchmark.
                Example: {"qa": "hotpotqa"}, {"math": "gsm8k"}, {"code": "mbpp"}.
            **strategy_kwargs (Dict[str, Any]): Additional keyword arguments to pass to the strategy's constructor.

        Returns:
            ReflexionCoTBaseStrategy: An instance of the appropriate ReflexionCoT strategy.

        Raises:
            ValueError: If the mode or benchmark is unsupported.
        """
        if "qa" in mode:
            if mode["qa"] == "hotpotqa":
                return ReflexionCoTHotQAStrategy(**strategy_kwargs)
            elif mode["qa"] == "triviaqa":
                return ReflexionCoTTriviaQAStrategy(**strategy_kwargs)
            elif mode["qa"] == "ambignq":
                return ReflexionCoTAmbigNQStrategy(**strategy_kwargs)
            elif mode["qa"] == "fever":
                return ReflexionCoTFEVERStrategy(**strategy_kwargs)
            else:
                raise ValueError(f"Unsupported QA benchmark: {mode['qa']}")
        elif "math" in mode:
            if mode["math"] == "gsm8k":
                return ReflexionCoTGSM8KStrategy(**strategy_kwargs)
            elif mode["math"] == "svamp":
                return ReflexionCoTSVAMPStrategy(**strategy_kwargs)
            elif mode["math"] == "tabmwp":
                return ReflexionCoTTabMWPStrategy(**strategy_kwargs)
            else:
                raise ValueError(f"Unsupported Math benchmark: {mode['math']}")
        elif "code" in mode:
            if mode["code"] == "humaneval":
                return ReflexionCoTHEvalStrategy(**strategy_kwargs)
            elif mode["code"] == "mbpp":
                return ReflexionCoTMBPPStrategy(**strategy_kwargs)
            else:
                raise ValueError(f"Unsupported Code benchmark: {mode['code']}")
        else:
            raise ValueError(f"Unsupported mode: {mode}")


class ReflexionReActStrategyFactory:
    """A factory class for creating instances of different ReflexionReAct strategies based on the specified mode and benchmark.

    Methods:
        get_strategy(mode: Dict[str, str], **strategy_kwargs) -> ReflexionReActBaseStrategy:
            Returns an instance of the appropriate ReflexionReAct strategy based on the provided mode and benchmark.
    """

    @staticmethod
    def get_strategy(
        mode: Dict[str, str], **strategy_kwargs: Any
    ) -> ReflexionReActBaseStrategy:
        """Returns an instance of the appropriate ReflexionReAct strategy based on the provided mode and benchmark.

        Available modes:
            - qa: "hotpotqa", "triviaqa", "ambignq", "fever"
            - math: "gsm8k", "svamp", "tabmwp"
            - code: "mbpp", "humaneval"

        Args:
            mode (Dict[str, str]): A dictionary specifying the mode and benchmark.
                Example: {"qa": "hotpotqa"}, {"math": "gsm8k"}, {"code": "mbpp"}.
            **strategy_kwargs (Dict[str, Any]): Additional keyword arguments to pass to the strategy's constructor.

        Returns:
            ReflexionReActBaseStrategy: An instance of the appropriate ReflexionReAct strategy.

        Raises:
            ValueError: If the mode or benchmark is unsupported.
        """
        if "qa" in mode:
            if mode["qa"] == "hotpotqa":
                return ReflexionReActHotQAStrategy(**strategy_kwargs)
            elif mode["qa"] == "triviaqa":
                return ReflexionReActTriviaQAStrategy(**strategy_kwargs)
            elif mode["qa"] == "ambignq":
                return ReflexionReActAmbigNQStrategy(**strategy_kwargs)
            elif mode["qa"] == "fever":
                return ReflexionReActFEVERStrategy(**strategy_kwargs)
            else:
                raise ValueError(f"Unsupported QA benchmark: {mode['qa']}")
        elif "math" in mode:
            if mode["math"] == "gsm8k":
                return ReflexionReActGSM8KStrategy(**strategy_kwargs)
            elif mode["math"] == "svamp":
                return ReflexionReActSVAMPStrategy(**strategy_kwargs)
            elif mode["math"] == "tabmwp":
                return ReflexionReActTabMWPStrategy(**strategy_kwargs)
            else:
                raise ValueError(f"Unsupported Math benchmark: {mode['math']}")
        elif "code" in mode:
            if mode["code"] == "humaneval":
                return ReflexionReActHEvalStrategy(**strategy_kwargs)
            elif mode["code"] == "mbpp":
                return ReflexionReActMBPPStrategy(**strategy_kwargs)
            else:
                raise ValueError(f"Unsupported Code benchmark: {mode['code']}")
        else:
            raise ValueError(f"Unsupported mode: {mode}")
