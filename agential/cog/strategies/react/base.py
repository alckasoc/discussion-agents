"""Base ReAct Agent strategy class."""

from abc import abstractmethod
from typing import Dict, Tuple

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel

from agential.cog.strategies.base import BaseStrategy


class ReActBaseStrategy(BaseStrategy):
    """An abstract base class for defining strategies for the ReAct Agent.

    Attributes:
        llm (BaseChatModel): The language model used for generating answers and critiques.
    """

    def __init__(self, llm: BaseChatModel) -> None:
        """Initialization."""
        super().__init__(llm)

    @abstractmethod
    def generate_action(
        self,
        question: str,
        examples: str,
        prompt: str,
        additional_keys: Dict[str, str],
    ) -> Tuple[str, str]:
        """Generates an action based on the question, examples, and prompt.

        Args:
            question (str): The question to be answered.
            examples (str): Examples to guide the generation process.
            prompt (str): The prompt used for generating the action.
            additional_keys (Dict[str, str]): Additional keys for the generation process.

        Returns:
            Tuple[str, str]: The generated action type and query.
        """
        pass

    @abstractmethod
    def generate_observation(self, idx: int, action_type: str, query: str) -> str:
        """Generates an observation based on the action type and query.

        Args:
            idx (int): The index of the observation.
            action_type (str): The type of action to be performed.
            query (str): The query for the action.

        Returns:
            str: The generated observation.
        """
        pass

    @abstractmethod
    def create_output_pydantic(
        self, thought: str, action_type: str, query: str, observation: str
    ) -> BaseModel:
        """Creates a Pydantic model of the output components.

        Args:
            thought (str): The generated thought.
            action_type (str): The type of action performed.
            query (str): The query for the action.
            observation (str): The generated observation.

        Returns:
            BaseModel: A Pydantic model containing the thought, action type, query, and observation.
        """
        pass

    @abstractmethod
    def create_output_dict(
        self, thought: str, action_type: str, query: str, obs: str
    ) -> Dict[str, str]:
        """Creates a dictionary of the output components.

        Args:
            thought (str): The generated thought.
            action_type (str): The type of action performed.
            query (str): The query for the action.
            obs (str): The generated observation.

        Returns:
            Dict[str, str]: A dictionary containing the thought, action type, query, and observation.
        """
        pass

    @abstractmethod
    def halting_condition(
        self,
        idx: int,
        question: str,
        examples: str,
        prompt: str,
        additional_keys: Dict[str, str],
    ) -> bool:
        """Determines whether the halting condition has been met.

        Args:
            idx (int): The current step index.
            question (str): The question being answered.
            examples (str): Examples to guide the generation process.
            prompt (str): The prompt used for generating the thought and action.
            additional_keys (Dict[str, str]): Additional keys for the generation process.

        Returns:
            bool: True if the halting condition is met, False otherwise.
        """
        pass
