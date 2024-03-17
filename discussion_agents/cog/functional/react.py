"""Functional module for ReAct."""
from langchain.prompts import PromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages.human import HumanMessage
from tiktoken import Encoding
from typing import Optional , Union , List

from discussion_agents.utils.parse import remove_newline



def _build_agent_prompt(question: str, scratchpad: str, examples: str , prompt_template: str) -> str:

    """Constructs a prompt template for the agent.

    This function formats a predefined prompt template (REACT_INSTRUCTION) with examples,
    the provided question, and a scratchpad.

    Args:
        question (str): The question to be included in the prompt.
        scratchpad (str): Additional scratchpad information to be included.
        examples (str): The example as a guide of how the test should be prompted.
        prompt_template (str): The template of the prompt that is inputted into scratchpad.

    Returns:
        str: A formatted prompt template ready for use.
    """

    prompt = PromptTemplate.from_template(prompt_template).format(
        examples=examples,
        question=question,
        scratchpad=scratchpad
    )
    return prompt


def _prompt_agent(
        llm: BaseChatModel, 
        question: str, 
        scratchpad: str, 
        examples: str, 
        prompt_template: str, 
        stop: str='\n'
    ) -> str:

    """Generates a response from the LLM based on a given question and scratchpad.

    This function creates a prompt using `_build_agent_prompt` and then gets the LLM's output.
    The newline characters in the output are removed before returning.

    Args:
        llm (BaseChatModel): The language model to be prompted.
        question (str): The question to ask the language model.
        scratchpad (str): Additional context or information for the language model.
        examples (str): The example used for specific benchmark for AI model to generate prompt accordingly.
        prompt_template (str): The template of the prompt that is inputted into scratchpad.
        stop (str): The stop condition for the language model. Defaults to None.

    Returns:
        str: The processed response from the language model.
    """
    prompt = _build_agent_prompt(
        question=question, 
        scratchpad=scratchpad, 
        examples=examples, 
        prompt_template=prompt_template
    )

    out = llm([HumanMessage(content=prompt)], stop=stop).content

    assert isinstance(out, str)
    return out


def _is_halted(
    finished: bool,
    step_n: int,
    max_steps: int,
    question: str,
    scratchpad: str,
    max_tokens: int,
    enc: Encoding,
    examples: str ,
    prompt_template: str
) -> bool:
    """Determines whether the agent's operation should be halted.

    This function checks if the operation should be halted based on three conditions:
    completion (finished), exceeding maximum steps, or exceeding maximum token limit.
    The token limit is evaluated based on the encoded length of the prompt.

    Args:
        finished (bool): Flag indicating if the operation is completed.
        step_n (int): Current step number.
        max_steps (int): Maximum allowed steps.
        question (str): The question being processed.
        scratchpad (str): The scratchpad content.
        max_tokens (int): Maximum allowed token count.
        enc (Encoding): The encoder to calculate token length.
        examples (str): The example of the output prompt.
        prompt_template (str): The template of the prompt.
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
                    prompt_template=prompt_template
                )
            )
        )
        > max_tokens
    )
    return finished or over_max_steps or over_token_limit


