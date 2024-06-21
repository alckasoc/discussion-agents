"""CRITIC Agent.

Original Paper: https://arxiv.org/pdf/2305.11738
Paper Repository: https://github.com/microsoft/ProphetNet/tree/master/CRITIC
"""

from typing import Any, Dict, List

from langchain_core.language_models.chat_models import BaseChatModel

from pydantic import BaseModel

from agential.cog.agent.base import BaseAgent
from agential.cog.strategies.strategy_factory import CriticStrategyFactory

class CriticPydanticOutput(BaseModel):
    class Config:
        title = 'Critic Output'
        description = 'Critic output for different modes'

    class QA(BaseModel):
        answer: str
        critique: str
        query: str
        search_result: str
        revised_answer: str

    class Math(BaseModel):
        code: Union[int, str]
        critique: str
        execution_status: str
        code_answer: Union[int, str]
        improved_code: Union[int, str]

    class Code(BaseModel):
        code: Union[int, str]
        critique: str
        execution_status: str
        improved_code: Union[int, str]

        answer: str = Field(..., description = "The answer generated by the agent.")
        critique : str = Field(..., description = "The critique of the answer generated by the agent.")
        query: str = Field(..., description = "The query requested by the agent. ")
        search_result: str = Field(..., description = "The search result requested by the agent.")
        revised_answer: str = Field(..., description = "The revised answer generated by the agent. ")

        code: int/str? = Field(..., description = "The code generated by the agent.")
        critique: str = Field(..., description = " The critique of the answer generated by the agent.")

        execution_status: str = Field(..., description = "The execution status of the agent.")
        code_answer: int/str? = Field(..., description = "The code answer generated by the agent.")
        improved_code: int/str? = Field(..., description = "The improved code generated by the agent.")





class CriticAgent(BaseAgent):
    """CRITIC Agent.
    Attributes:
        llm (BaseChatModel): An instance of a language model used for generating initial answers
            and critiques.
        mode (Dict[str, str]): A dictionary specifying the CRITIC agent's mode and the benchmark.
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
        self.strategy = CriticStrategyFactory().get_strategy(
            mode=self.mode, llm=self.llm, **strategy_kwargs
        )
    def generate(
        self,
        question: str,
        examples: str,
        prompt: str,
        critique_examples: str,
        critique_prompt: str,
        additional_keys: Dict[str, str] = {},
        critique_additional_keys: Dict[str, str] = {},
        max_interactions: int = 7,
        use_tool: bool = True,
        reset: bool = True,
        **kwargs: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Generates an answer that is refined with search results.
@@ -79,26 +131,30 @@
        out = []
        # Initial answer generation.
        answer = self.strategy.generate(question, examples, prompt, additional_keys)
        critique = ""
        for idx in range(max_interactions):
            critique, external_tool_info = self.strategy.generate_critique(
                idx=idx,
                question=question,
                examples=critique_examples,
                answer=answer,
                critique=critique,
                prompt=critique_prompt,
                additional_keys=critique_additional_keys,
                use_tool=use_tool,
                max_interactions=max_interactions,
                **kwargs,
            )

            out.append(
                self.strategy.create_output_dict(answer, critique, external_tool_info)

                self.strategy.create_output_pydantic(
                    XXXXXXX
                )
            )

            if self.strategy.halting_condition():
                break

            # Update answer for the next iteration.
            answer = self.strategy.update_answer_based_on_critique(
                question=question,
                examples=critique_examples,
                answer=answer,
                critique=critique,
                prompt=critique_prompt,
                additional_keys=critique_additional_keys,
                external_tool_info=external_tool_info,
                **kwargs,
            )

        return out

    def reset(self) -> None:
        """Resets the CRITIC Agent's internal state."""
        self.strategy.reset()
