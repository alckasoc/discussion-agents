"""Base CRITIC Agent strategy class."""

from abc import abstractmethod
from typing import Dict

from agential.cog.strategies.base import BaseStrategy


class CriticBaseStrategy(BaseStrategy):
    """An abstract base class for defining strategies for the CRITIC Agent."""

    @abstractmethod
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
        **kwargs,
    ) -> str:
        """Generates a critique of the provided answer using the given language model, question, examples, and prompt.

        Args:
            idx (int): The index of the current interaction.
            question (str): The question that was answered by the language model.
            examples (str): Few-shot examples to guide the language model in generating the critique.
            answer (str): The answer to be critiqued.
            critique (str): The previous critique, if any.
            prompt (str): The instruction template used to prompt the language model for the critique.
            additional_keys (Dict[str, str]): Additional keys to format the critique prompt.
            use_tool (bool): Whether to use an external tool (e.g., code interpreter, search tool) during critique.
            max_interactions (int): The maximum number of critique interactions.
            **kwargs: Additional arguments that might be needed for specific implementations.

        Returns:
            str: The generated critique.
        """
        pass

    @abstractmethod
    def create_output_dict(
        self, answer: str, critique: str, external_tool_info: Dict[str, str]
    ) -> Dict[str, str]:
        """Creates a dictionary containing the answer and critique, along with any additional key updates.

        Args:
            answer (str): The original answer.
            critique (str): The generated critique.
            external_tool_info (Dict[str, str]): Information from any external tools used during the critique.

        Returns:
            Dict[str, str]: A dictionary containing the answer, critique, and additional key updates.
        """
        pass

    @abstractmethod
    def update_answer_based_on_critique(
        self,
        question: str,
        examples: str,
        answer: str,
        critique: str,
        prompt: str,
        additional_keys: Dict[str, str],
        external_tool_info: Dict[str, str],
        **kwargs,
    ) -> str:
        """Updates the answer based on the provided critique using the given language model and question.

        Args:
            question (str): The question that was answered by the language model.
            examples (str): Few-shot examples to guide the language model in generating the updated answer.
            answer (str): The original answer to be updated.
            critique (str): The critique of the original answer.
            prompt (str): The instruction template used to prompt the language model for the update.
            additional_keys (Dict[str, str]): Additional keys to format the update prompt.
            external_tool_info (Dict[str, str]): Information from any external tools used during the critique.
            **kwargs: Additional arguments that might be needed for specific implementations.

        Returns:
            str: The updated answer.
        """
        pass

    @abstractmethod
    def halting_condition(self, critique: str) -> bool:
        """Determines whether the critique meets the halting condition for stopping further updates.

        Args:
            critique (str): The generated critique.

        Returns:
            bool: True if the halting condition is met, False otherwise.
        """
        pass