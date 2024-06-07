"""CRITIC Agent strategies for Math."""

from typing import Any, Dict, List, Tuple

from langchain_core.language_models.chat_models import BaseChatModel

from agential.cog.functional.critic import _prompt_agent, _prompt_critique
from agential.cog.strategies.critic.base import CriticBaseStrategy
from agential.utils.general import safe_execute
from agential.utils.validation import validate_overlapping_keys


class CriticMathStrategy(CriticBaseStrategy):
    """A strategy class for Math benchmarks using the CRITIC agent.

    Attributes:
        llm (BaseChatModel): The language model used for generating answers and critiques.
        patience (int): The number of interactions to tolerate the same incorrect answer
            before halting further attempts. Defaults to 2.
    """

    def __init__(self, llm: BaseChatModel, patience: int = 2) -> None:
        """Initialization."""
        super().__init__(llm)
        self.patience = patience
        self._answer_history: List[Dict[str, Any]] = []
        self._prev_code_answer = ""
        self.patience_counter = 0
        self._halt = False

    def generate(
        self,
        question: str,
        examples: str,
        prompt: str,
        additional_keys: Dict[str, str],
        **kwargs: Dict[str, Any],
    ) -> str:
        """Generates an answer for the given question using the provided prompt and examples.

        Args:
            question (str): The math question to generate an answer for.
            examples (str): Few-shot examples to guide the language model.
            prompt (str): The prompt to generate an answer.
            additional_keys (Dict[str, str]): Additional keys for the prompt.
            **kwargs (Dict[str, Any]): Additional arguments.

        Returns:
            str: The generated answer.
        """
        answer = _prompt_agent(
            llm=self.llm,
            question=question,
            examples=examples,
            additional_keys=additional_keys,
            prompt=prompt,
        )
        answer = answer.split("```python")[-1].split("```")[0].strip()

        return answer

    def generate_critique(
        self,
        idx: int,
        question: str,
        examples: str,
        answer: str,
        critique: str,
        prompt: str,
        additional_keys: Dict[str, str],
        use_tool: bool,
        max_interactions: int,
        **kwargs: Dict[str, Any],
    ) -> Tuple[str, Dict[str, Any]]:
        """Generates a critique for the provided answer using the given prompt and examples.

        This method does the following:
            1. If `use_tool` is True, execute the answer as code and store the result.
            2. Update the answer history with the current answer and external tool info.
            3. Check if the current code answer is the same as the previous one:
               - If yes, increment the patience counter.
               - If the patience counter reaches the patience limit, set the halt flag.
               - Otherwise, update the previous code answer.
            4. Find the last valid answer from the history that includes external tool info.
            5. Validate and merge additional keys with external tool info.
            6. Generate a new critique using the updated answer and keys.
            7. Return the new critique and external tool info.

        Args:
            idx (int): The index of the current interaction.
            question (str): The math question that was answered.
            examples (str): Few-shot examples to guide the critique.
            answer (str): The answer to critique.
            critique (str): Existing critique to build upon.
            prompt (str): The prompt to generate a critique.
            additional_keys (Dict[str, str]): Additional keys for the prompt.
            use_tool (bool): Whether to use an external tool during critique.
            max_interactions (int): The maximum number of interactions allowed.
            **kwargs (Dict[str, Any]): Additional arguments for specific implementations.

        Returns:
            Tuple[str, Dict[str, Any]]: The generated critique and external tool information.
        """
        external_tool_info = {}
        if use_tool:
            code_answer, execution_status = safe_execute(answer)
            external_tool_info = {
                "execution_status": execution_status,
                "code_answer": code_answer[0] if code_answer[0] is not None else "",
            }
            self._answer_history.append(
                {"answer": answer, "external_tool_info": external_tool_info}
            )

            if code_answer[0] == self._prev_code_answer:
                self.patience_counter += 1
                if self.patience_counter == self.patience:
                    self._halt = True
            else:
                self._prev_code_answer = code_answer[0]

            last_valid_idx = -1
            for i in range(len(self._answer_history) - 1, -1, -1):
                if (
                    self._answer_history[i]["external_tool_info"]["code_answer"]
                    is not None
                ):  # type: ignore
                    last_valid_idx = i
                    break

            external_tool_info = self._answer_history[last_valid_idx][
                "external_tool_info"
            ]  # type: ignore
            answer = self._answer_history[last_valid_idx]["answer"]  # type: ignore

            validate_overlapping_keys(additional_keys, external_tool_info)

        additional_keys = additional_keys.copy()
        additional_keys.update(external_tool_info)

        new_critique = _prompt_critique(
            llm=self.llm,
            question=question,
            examples=examples,
            answer=answer,
            critique="",
            additional_keys=additional_keys,
            prompt=prompt,
        ).split("Here's")[0]

        return new_critique, external_tool_info

    def create_output_dict(
        self, answer: str, critique: str, external_tool_info: Dict[str, str]
    ) -> Dict[str, str]:
        """Creates an output dictionary containing the answer, critique, and external tool information.

        Args:
            answer (str): The generated answer.
            critique (str): The generated critique.
            external_tool_info (Dict[str, str]): Information from external tool execution.

        Returns:
            Dict[str, str]: The output dictionary.
        """
        output_dict = {"code": answer, "critique": critique, **external_tool_info}
        return output_dict

    def update_answer_based_on_critique(
        self,
        question: str,
        examples: str,
        answer: str,
        critique: str,
        prompt: str,
        additional_keys: Dict[str, str],
        external_tool_info: Dict[str, str],
        **kwargs: Dict[str, Any],
    ) -> str:
        """Updates the answer based on the given critique.

        Args:
            question: The question that was answered by the language model.
            examples: Few-shot examples to guide the language model.
            answer: The answer provided by the language model.
            critique: The critique of the answer.
            prompt: The prompt to be used for generating the updated answer.
            additional_keys: Additional context or parameters to include in the critique prompt.
            external_tool_info: Information from any external tool used.
            **kwargs (Dict[str, Any]): Additional parameters for flexibility.

        Returns:
            str: The updated answer.
        """
        validate_overlapping_keys(additional_keys, external_tool_info)
        additional_keys = additional_keys.copy()
        additional_keys.update(external_tool_info)

        new_answer = _prompt_critique(
            llm=self.llm,
            question=question,
            examples=examples,
            answer=answer,
            critique=f"{critique}\n\nHere's a better solution:\n```python\n",
            additional_keys=additional_keys,
            prompt=prompt,
        )
        new_answer = new_answer.split("```python")[-1].split("```")[0].strip()

        return new_answer

    def halting_condition(self) -> bool:
        """Checks if the halting condition has been met.

        Returns True if the CRITIC Agent's generated answer remains the same for `patience` number of steps.

        Returns:
            bool: True if the halting condition has been met, False otherwise.
        """
        return self._halt

    def reset(self) -> None:
        """Resets the strategy to its initial state.

        Resets internal variables keeping track of halting and answer history.

        Returns:
            bool: True if the reset was successful, False otherwise.
        """
        self._answer_history = []
        self._prev_code_answer = ""
        self.patience_counter = 0
        self._halt = False


class CritGSM8KStrategy(CriticMathStrategy):
    """A strategy class for the GSM8K benchmark using the CRITIC agent."""

    pass


class CritSVAMPStrategy(CriticMathStrategy):
    """A strategy class for the SVAMP benchmark using the CRITIC agent."""

    pass


class CritTabMWPStrategy(CriticMathStrategy):
    """A strategy class for the TabMWP benchmark using the CRITIC agent."""

    pass
