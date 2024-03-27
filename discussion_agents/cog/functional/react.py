"""Functional module for ReAct."""
from langchain.prompts import PromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages.human import HumanMessage
from tiktoken import Encoding

from discussion_agents.cog.prompts.react import (
    REACT_INSTRUCTION,
)
from discussion_agents.utils.parse import remove_newline


def _build_agent_prompt(
    question: str,
    scratchpad: str,
    examples: str,
    max_steps: int,
    prompt: str = REACT_INSTRUCTION,
) -> str:
    """Constructs a prompt template for the agent.

    This function formats a prompt template string with examples,
    the provided question, a scratchpad, and max steps.

    Args:
        question (str): The question to be included in the prompt.
        scratchpad (str): Additional scratchpad information to be included.
        examples (str): Fewshot examples.
        max_steps (int): Max number of steps.
        prompt (str, optional): Prompt template string. Defaults to REACT_INSTRUCTION. Must include question,
            scratchpad, examples, and max_steps.

    Returns:
        str: A formatted prompt template ready for use.
    """
    prompt = PromptTemplate.from_template(prompt).format(
        question=question,
        scratchpad=scratchpad,
        examples=examples,
        max_steps=max_steps,
    )
    return prompt


def _prompt_agent(
    llm: BaseChatModel,
    question: str,
    scratchpad: str,
    examples: str,
    max_steps: int,
    prompt: str = REACT_INSTRUCTION,
) -> str:
    """Generates a response from the LLM based on a given question and scratchpad.

    This function creates a prompt using `_build_agent_prompt` and then gets the LLM's
    output. The newline characters in the output are removed before returning.

    Args:
        llm (BaseChatModel): The language model to be prompted.
        question (str): The question to ask the language model.
        scratchpad (str): Additional context or information for the language model.
        examples (str): Fewshot examples.
        max_steps (int): Maximum number of steps.
        prompt (str, optional): Prompt template string. Defaults to REACT_INSTRUCTION. Must include question,
            scratchpad, examples, and max_steps.

    Returns:
        str: The processed response from the language model.
    """
    prompt = _build_agent_prompt(
        question=question,
        scratchpad=scratchpad,
        examples=examples,
        max_steps=max_steps,
        prompt=prompt,
    )
    out = llm(
        [
            HumanMessage(
                content=prompt,
            )
        ]
    ).content
    assert isinstance(out, str)
    return remove_newline(out)


def _is_halted(
    finished: bool,
    step_n: int,
    question: str,
    scratchpad: str,
    examples: str,
    max_steps: int,
    max_tokens: int,
    enc: Encoding,
    prompt: str = REACT_INSTRUCTION,
) -> bool:
    """Determines whether the agent's operation should be halted.

    This function checks if the operation should be halted based on three conditions:
    completion (finished), exceeding maximum steps, or exceeding maximum token limit.
    The token limit is evaluated based on the encoded length of the prompt.

    Args:
        finished (bool): Flag indicating if the operation is completed.
        step_n (int): Current step number.
        question (str): The question being processed.
        scratchpad (str): The scratchpad content.
        examples (str): Fewshot examples.
        max_steps (int): Maximum allowed steps.
        max_tokens (int): Maximum allowed token count.
        enc (Encoding): The encoder to calculate token length.
        prompt (str, optional): Prompt template string. Defaults to REACT_INSTRUCTION. Must include question,
            scratchpad, examples, and max_steps.

    Returns:
        bool: True if the operation should be halted, False otherwise.
    """
    over_max_steps = step_n > max_steps
    over_token_limit = (
        len(
            enc.encode(
                _build_agent_prompt(
                    question=question,
                    scratchpad=scratchpad,
                    examples=examples,
                    max_steps=max_steps,
                    prompt=prompt,
                )
            )
        )
        > max_tokens
    )
    return finished or over_max_steps or over_token_limit
