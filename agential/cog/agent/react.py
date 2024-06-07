"""ReAct Agent.

Original Paper: https://arxiv.org/abs/2210.03629
Paper Repository: https://github.com/ysymyth/ReAct
"""

from typing import Any, Dict, List

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel

from agential.cog.agent.base import BaseAgent
from agential.cog.strategies.strategy_factory import ReActStrategyFactory


class ReActAgent(BaseAgent):
    """ReAct agent from the original paper.

    Attributes:
        llm (BaseChatModel): An instance of a language model used for generating initial answers
            and critiques.
        mode (Dict[str, str]): A dictionary specifying the ReAct agent's mode and the benchmark.
            For example, {"qa": "hotpotqa"}, {"math": "gsm8k"}, or {"code": "mbpp"}.
        **strategy_kwargs (Dict[str, Any]): Additional strategy-specific arguments.
    """

    def __init__(
        self,
        llm: BaseChatModel,
        mode: Dict[str, str],
        **strategy_kwargs: Dict[str, Any],
    ) -> None:
        """Initialization."""
        super().__init__()
        self.llm = llm
        self.mode = mode

        self.strategy = ReActStrategyFactory().get_strategy(
            mode=self.mode, llm=self.llm, **strategy_kwargs
        )

    def generate(
        self,
        question: str,
        examples: str,
        prompt: str,
        additional_keys: Dict[str, str] = {},
        reset: bool = True,
        **kwargs: Dict[str, Any],
    ) -> List[BaseModel]:
        """Processes a given question through ReAct.

        Iteratively applies the think-act-observe cycle to generate an answer for the question.
        The process continues until the operation is halted based on certain conditions.

        Args:
            question (str): The question to be processed.
            examples (str, optional): Fewshot examples.
            prompt (str, optional): Prompt template string.
            additional_keys (Dict[str, str]): Additional keys to format the prompt. Defaults to {}.
            reset (bool, optional): Whether to reset the internal state before processing. Defaults to True.
            **kwargs (Dict[str, Any]): Additional parameters for flexibility.

        Returns:
            List[ReActOutput]: The list of accumulated output from the ReAct process,
                each step represented by a ReActOutput model.
        """
        if reset:
            self.reset()

        idx = 1
        out = []
        while not self.strategy.halting_condition(
            idx=idx,
            question=question,
            examples=examples,
            additional_keys=additional_keys,
            prompt=prompt,
            **kwargs,
        ):
            # Think.
            thought = self.strategy.generate(
                question=question,
                examples=examples,
                additional_keys=additional_keys,
                prompt=prompt,
                **kwargs,
            )

            # Act.
            action_type, query = self.strategy.generate_action(
                question=question,
                examples=examples,
                additional_keys=additional_keys,
                prompt=prompt,
                **kwargs,
            )

            # Observe.
            obs = self.strategy.generate_observation(
                idx=idx, action_type=action_type, query=query
            )
            # This line is modified for testing structured output in Pydantic model
            out.append(
                self.strategy.create_output_pydantic(
                    thought=thought,
                    action_type=action_type,
                    query=query,
                    observation=obs,
                )
            )

            idx += 1

        return out

    def reset(self) -> None:
        """Resets the internal state of the ReAct agent.

        Sets the step number, finished flag, and scratchpad to their initial values.
        """
        self.strategy.reset()
