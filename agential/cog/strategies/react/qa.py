"""ReAct Agent strategies for QA."""

import re

from typing import Any, Dict, Tuple

import tiktoken

from langchain_community.docstore.wikipedia import Wikipedia
from langchain_core.language_models.chat_models import BaseChatModel
from tiktoken.core import Encoding

from agential.cog.functional.react import _is_halted, _prompt_agent
from agential.cog.strategies.react.base import ReActBaseStrategy
from agential.utils.docstore import DocstoreExplorer
from agential.utils.parse import remove_newline


def parse_qa_action(string: str) -> Tuple[str, str]:
    """Parses an action string into an action type and its argument.

    This method is used in ReAct.

    Args:
        string (str): The action string to be parsed.

    Returns:
        Tuple[str, str]: A tuple containing the action type and argument.
    """
    pattern = r"^(\w+)\[(.+)\]$"
    match = re.match(pattern, string)

    if match:
        action_type = match.group(1)
        argument = match.group(2)
    else:
        action_type = ""
        argument = ""
    return action_type, argument


class ReActQAStrategy(ReActBaseStrategy):
    """A strategy class for QA benchmarks using the ReAct agent.

    Attributes:
        llm (BaseChatModel): The language model used for generating answers and critiques.
        max_steps (int): The maximum number of steps the agent can take.
        max_tokens (int): The maximum number of tokens allowed for a response.
        docstore (DocstoreExplorer): The document store used for searching and looking up information.
        enc (Encoding): The encoding used for the language model.
    """

    def __init__(
        self,
        llm: BaseChatModel,
        max_steps: int = 6,
        max_tokens: int = 5000,
        docstore: DocstoreExplorer = DocstoreExplorer(Wikipedia()),
        enc: Encoding = tiktoken.encoding_for_model("gpt-3.5-turbo"),
    ) -> None:
        """Initialization."""
        super().__init__(llm)
        self.max_steps = max_steps
        self.max_tokens = max_tokens
        self.docstore = docstore
        self.enc = enc

        self._scratchpad = ""
        self._answer = ""
        self._finished = False

    def generate(
        self,
        question: str,
        examples: str,
        prompt: str,
        additional_keys: Dict[str, str],
        **kwargs: Any,
    ) -> str:
        """Generates a thought based on the question, examples, and prompt.

        Args:
            question (str): The question to be answered.
            examples (str): Examples to guide the generation process.
            prompt (str): The prompt used for generating the thought.
            additional_keys (Dict[str, str]): Additional keys for the generation process.
            **kwargs (Any): Additional arguments.

        Returns:
            str: The generated thought.
        """
        max_steps = kwargs.get("max_steps", self.max_steps)  # type: ignore

        self._scratchpad += "\nThought:"
        thought = _prompt_agent(
            llm=self.llm,
            question=question,
            scratchpad=self._scratchpad,
            examples=examples,
            max_steps=max_steps,  # type: ignore
            prompt=prompt,
            additional_keys=additional_keys,
        )
        thought = remove_newline(thought).split("Action")[0].strip()
        self._scratchpad += " " + thought

        return thought

    def generate_action(
        self,
        question: str,
        examples: str,
        prompt: str,
        additional_keys: Dict[str, str],
        **kwargs: Any,
    ) -> Tuple[str, str]:
        """Generates an action based on the question, examples, and prompt.

        Args:
            question (str): The question to be answered.
            examples (str): Examples to guide the generation process.
            prompt (str): The prompt used for generating the action.
            additional_keys (Dict[str, str]): Additional keys for the generation process.
            **kwargs (Any): Additional arguments.

        Returns:
            Tuple[str, str]: The generated action type and query.
        """
        max_steps = kwargs.get("max_steps", self.max_steps)
        self._scratchpad += "\nAction:"
        action = _prompt_agent(
            llm=self.llm,
            question=question,
            scratchpad=self._scratchpad,
            examples=examples,
            max_steps=max_steps,  # type: ignore
            prompt=prompt,
            additional_keys=additional_keys,
        )
        action = remove_newline(action).split("Observation")[0]
        self._scratchpad += " " + action
        action_type, query = parse_qa_action(action)

        return action_type, query

    def generate_observation(
        self, idx: int, action_type: str, query: str
    ) -> Tuple[str, Dict[str, Any]]:
        """Generates an observation based on the action type and query.

        Args:
            idx (int): The index of the observation.
            action_type (str): The type of action to be performed.
            query (str): The query for the action.

        Returns:
            Tuple[str, Dict[str, Any]]: The generated observation and external tool outputs.
        """
        external_tool_info = {"search_result": "", "lookup_result": ""}

        self._scratchpad += f"\nObservation {idx}: "
        if action_type.lower() == "finish":
            self._answer = query
            self._finished = True
            obs = query
        elif action_type.lower() == "search":
            try:
                search_result = self.docstore.search(query)
                external_tool_info["search_result"] = search_result
                obs = remove_newline(search_result)
            except Exception:
                obs = "Could not find that page, please try again."
        elif action_type.lower() == "lookup":
            try:
                lookup_result = self.docstore.lookup(query)
                external_tool_info["lookup_result"] = lookup_result
                obs = remove_newline(lookup_result)

            except ValueError:
                obs = "The last page Searched was not found, so you cannot Lookup a keyword in it. Please try one of the similar pages given."
        else:
            obs = "Invalid Action. Valid Actions are Lookup[<topic>] Search[<topic>] and Finish[<answer>]."
        self._scratchpad += obs

        return obs, external_tool_info

    def create_output_dict(
        self,
        thought: str,
        action_type: str,
        query: str,
        obs: str,
        external_tool_info: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Creates a dictionary of the output components.

        Args:
            thought (str): The generated thought.
            action_type (str): The type of action performed.
            query (str): The query for the action.
            obs (str): The generated observation.
            external_tool_info (Dict[str, Any]): The external tool outputs.

        Returns:
            Dict[str, Any]: A dictionary containing the thought, action type, query, observation, answer, and external tool output.
        """
        return {
            "thought": thought,
            "action_type": action_type,
            "query": query,
            "observation": obs,
            "answer": self._answer,
            "external_tool_info": external_tool_info,
        }

    def halting_condition(
        self,
        idx: int,
        question: str,
        examples: str,
        prompt: str,
        additional_keys: Dict[str, str],
        **kwargs: Any,
    ) -> bool:
        """Determines whether the halting condition has been met.

        Args:
            idx (int): The current step index.
            question (str): The question being answered.
            examples (str): Examples to guide the generation process.
            prompt (str): The prompt used for generating the thought and action.
            additional_keys (Dict[str, str]): Additional keys for the generation process.
            **kwargs (Any): Additional arguments.

        Returns:
            bool: True if the halting condition is met, False otherwise.
        """
        max_steps = kwargs.get("max_steps", self.max_steps)

        return _is_halted(
            finished=self._finished,
            idx=idx,
            question=question,
            scratchpad=self._scratchpad,
            examples=examples,
            max_steps=max_steps,  # type: ignore
            max_tokens=self.max_tokens,
            enc=self.enc,
            prompt=prompt,
            additional_keys=additional_keys,
        )

    def reset(self, **kwargs: Any) -> None:
        """Resets the internal state of the strategy.

        Resets the scratchpad and the finished flag.

        Args:
            **kwargs (Any): Additional arguments.

        Returns:
            None
        """
        self._scratchpad = ""
        self._finished = False


class ReActHotQAStrategy(ReActQAStrategy):
    """A strategy class for the HotpotQA benchmark using the ReAct agent."""

    pass


class ReActTriviaQAStrategy(ReActQAStrategy):
    """A strategy class for the TriviaQA benchmark using the ReAct agent."""

    pass


class ReActAmbigNQStrategy(ReActQAStrategy):
    """A strategy class for the AmbigNQ benchmark using the ReAct agent."""

    pass


class ReActFEVERStrategy(ReActQAStrategy):
    """A strategy class for the FEVER benchmark using the ReAct agent."""

    pass
