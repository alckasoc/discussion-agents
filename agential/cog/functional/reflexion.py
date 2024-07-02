"""Functional module for Reflexion."""

from typing import Dict, List

import tiktoken

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages.human import HumanMessage
from langchain_core.prompts.prompt import PromptTemplate
from tiktoken.core import Encoding

from agential.cog.prompts.agent.reflexion import (
    LAST_TRIAL_HEADER,
    REFLECTION_HEADER,
)
from agential.utils.parse import remove_newline

gpt3_5_turbo_enc = tiktoken.encoding_for_model(
    "gpt-3.5-turbo"
)  # https://openai.com/blog/gpt-4-api-general-availability


def _truncate_scratchpad(
    scratchpad: str, n_tokens: int = 1600, tokenizer: Encoding = gpt3_5_turbo_enc
) -> str:
    """Truncates the scratchpad content to fit within a specified token limit.

    This function splits the scratchpad content into lines, filters out lines starting with 'Observation',
    and sorts them by token count. It then truncates the observations if the total token count exceeds the limit.

    Args:
        scratchpad (str): The scratchpad content to be truncated.
        n_tokens (int, optional): The maximum number of tokens allowed. Defaults to 1600.
        tokenizer (Encoding, optional): The tiktoken tokenizer used for counting tokens. Defaults to tiktoken's "gpt-3.5-turbo".

    Returns:
        str: The truncated scratchpad content.
    """
    # Split the scratchpad content into lines.
    lines = scratchpad.split("\n")
    # Filter out lines starting with 'Observation'.
    observations = filter(lambda x: x.startswith("Observation"), lines)
    # Sort observations by token count.
    observations_by_tokens = sorted(
        observations, key=lambda x: len(tokenizer.encode(x))
    )
    # Truncate observations if total token count exceeds limit.
    while len(tokenizer.encode("\n".join(lines))) > n_tokens:
        largest_observation = observations_by_tokens.pop(-1)
        ind = lines.index(largest_observation)
        # Replace the largest observation with a truncated message.
        lines[ind] = (
            largest_observation.split(":")[0] + ": [truncated wikipedia excerpt]"
        )
    return "\n".join(lines)


def _format_reflections(reflections: List[str], header: str = REFLECTION_HEADER) -> str:
    """Formats a list of reflection strings into a single formatted string.

    Args:
        reflections (List[str]): A list of reflection strings to be formatted.
        header (str, optional): A header to prepend to the formatted reflections. Defaults to REFLECTION_HEADER.

    Returns:
        str: The formatted string of reflections.
    """
    # Return formatted reflections if not empty.
    if reflections:
        return (
            header + "Reflections:\n- " + "\n- ".join([r.strip() for r in reflections])
        )
    else:
        return ""


def _format_last_attempt(
    question: str,
    scratchpad: str,
    header: str = LAST_TRIAL_HEADER,
    tokenizer: Encoding = gpt3_5_turbo_enc,
) -> str:
    """Formats the last attempt using the provided question and scratchpad content.

    Args:
        question (str): The question associated with the last attempt.
        scratchpad (str): The scratchpad content of the last attempt.
        header (str, optional): A header to prepend to the formatted last attempt. Defaults to LAST_TRIAL_HEADER.
        tokenizer (Encoding, optional): The tokenizer used for processing the scratchpad. Defaults to gpt3_5_turbo_enc.

    Returns:
        str: The formatted last attempt.
    """
    # Format the last attempt using the provided question and scratchpad.
    return (
        header
        + f"Question: {question}\n"
        + _truncate_scratchpad(scratchpad, tokenizer=tokenizer).strip("\n").strip()
        + "\n(END PREVIOUS TRIAL)\n"
    )


def _build_cot_agent_prompt(
    examples: str,
    reflections: str,
    question: str,
    scratchpad: str,
    prompt: str,
    additional_keys: Dict[str, str] = {},
) -> str:
    """Constructs a ReflexionCoT prompt template for the agent.

    This function formats a predefined prompt template (REFLEXION_COT_INSTRUCTION or
    REFLEXION_COT_INSTRUCTION_NO_CONTEXT) with examples,
    the provided question, and a scratchpad.

    Args:
        examples (str): Example inputs for the prompt template.
        reflections (List[str]): Existing list of reflections.
        question (str): The question being addressed.
        scratchpad (str): The scratchpad content related to the question.
        prompt (str, optional): Prompt template string.
        additional_keys (Dict[str, str]): Additional keys to format the prompt. Defaults to {}.

    Returns:
        str: A formatted prompt template ready for use.
    """
    prompt = PromptTemplate(
        input_variables=[
            "examples",
            "reflections",
            "question",
            "scratchpad",
        ]
        + list(additional_keys.keys()),
        template=prompt,
    ).format(
        examples=examples,
        reflections=reflections,
        question=question,
        scratchpad=scratchpad,
        **additional_keys,
    )

    return prompt


def _prompt_cot_agent(
    llm: BaseChatModel,
    examples: str,
    reflections: str,
    question: str,
    scratchpad: str,
    prompt: str,
    additional_keys: Dict[str, str] = {},
) -> str:
    """Generates a CoT prompt for thought and action.

    Used with ReflexionCoT.

    Args:
        llm (BaseChatModel): The language model to be used for generating the reflection.
        examples (str): Example inputs for the prompt template.
        reflections (List[str]): Existing list of reflections.
        question (str): The question being addressed.
        scratchpad (str): The scratchpad content related to the question.
        prompt (str, optional): Prompt template string.
        additional_keys (Dict[str, str]): Additional keys to format the prompt. Defaults to {}.

    Returns:
        str: The generated reflection prompt.
    """
    prompt = _build_cot_agent_prompt(
        examples=examples,
        reflections=reflections,
        question=question,
        scratchpad=scratchpad,
        prompt=prompt,
        additional_keys=additional_keys,
    )
    print(
        "<PROMPT AGENT===============================================================>"
    )
    print(prompt)
    print(
        "<PROMPT AGENT===============================================================>"
    )
    out = llm(
        [
            HumanMessage(
                content=prompt,
            )
        ]
    ).content
    print("<OUT AGENT===============================================================>")
    print(repr(out))
    print("<OUT AGENT===============================================================>")
    assert isinstance(out, str)
    return out


def _build_cot_reflection_prompt(
    examples: str,
    question: str,
    scratchpad: str,
    prompt: str,
    additional_keys: Dict[str, str] = {},
) -> str:
    """Constructs a ReflexionCoT prompt template for reflection.

    Args:
        examples (str): Example inputs for the prompt template.
        question (str): The question being addressed.
        scratchpad (str): The scratchpad content related to the question.
        prompt (str, optional): Prompt template string.
        additional_keys (Dict[str, str]): Additional keys to format the prompt. Defaults to {}.

    Returns:
        str: A formatted prompt template ready for use.
    """
    prompt = PromptTemplate(
        input_variables=["examples", "question", "scratchpad"]
        + list(additional_keys.keys()),
        template=prompt,
    ).format(
        examples=examples,
        question=question,
        scratchpad=scratchpad,
        **additional_keys,
    )

    return prompt


def _prompt_cot_reflection(
    llm: BaseChatModel,
    examples: str,
    question: str,
    scratchpad: str,
    prompt: str,
    additional_keys: Dict[str, str] = {},
) -> str:
    """Generates a reflection prompt.

    Used with ReflexionCoT.

    Args:
        llm (BaseChatModel): The language model to be used for generating the reflection.
        examples (str): Example inputs for the prompt template.
        question (str): The question being addressed.
        scratchpad (str): The scratchpad content related to the question.
        prompt (str, optional): Prompt template string.
        additional_keys (Dict[str, str]): Additional keys to format the prompt. Defaults to {}.

    Returns:
        str: The generated reflection prompt.
    """
    prompt = _build_cot_reflection_prompt(
        examples=examples,
        question=question,
        scratchpad=scratchpad,
        prompt=prompt,
        additional_keys=additional_keys,
    )
    print(
        "<PROMPT REFLECT===============================================================>"
    )
    print(prompt)
    print(
        "<PROMPT REFLECT===============================================================>"
    )
    out = llm(
        [
            HumanMessage(
                content=prompt,
            )
        ]
    ).content
    print(
        "<OUT REFLECT===============================================================>"
    )
    print(repr(out))
    print(
        "<OUT REFLECT===============================================================>"
    )
    assert isinstance(out, str)
    return out


def cot_reflect_last_attempt(scratchpad: str) -> List[str]:
    """Performs a reflection based on the last attempt (scratchpad).

    Used with ReflexionCoT.

    Args:
        question (str): The question associated with the last attempt.
        scratchpad (str): The scratchpad content from the last attempt.

    Returns:
        List[str]: A list with the scratchpad content.
    """
    return [scratchpad]


def cot_reflect_reflexion(
    llm: BaseChatModel,
    reflections: List[str],
    examples: str,
    question: str,
    scratchpad: str,
    prompt: str,
    additional_keys: Dict[str, str] = {},
) -> List[str]:
    """Perform reflexion-based reflecting.

    Used with ReflexionCoT. This function uses a language model to generate a new reflection based on the provided context, question,
    and scratchpad. The new reflection is added to the existing list of reflections.

    Args:
        llm (BaseChatModel): The language model used for generating the reflection.
        reflections (List[str]): Existing list of reflections.
        examples (str): Example inputs for the prompt template.
        question (str): The question being addressed.
        scratchpad (str): The scratchpad content related to the question.
        prompt (str, optional): Prompt template string.
        additional_keys (Dict[str, str]): Additional keys to format the prompt. Defaults to {}

    Returns:
        List[str]: An updated list of reflections.
    """
    new_reflection = _prompt_cot_reflection(
        llm=llm,
        examples=examples,
        question=question,
        scratchpad=scratchpad,
        prompt=prompt,
        additional_keys=additional_keys,
    )
    new_reflection = remove_newline(new_reflection)
    reflections += [new_reflection]
    return reflections


def cot_reflect_last_attempt_and_reflexion(
    llm: BaseChatModel,
    examples: str,
    question: str,
    scratchpad: str,
    prompt: str,
    additional_keys: Dict[str, str] = {},
) -> List[str]:
    """Performs reflection with the reflection of the last attempt and reflexion.

    Used with ReflexionCoT.

    Args:
        llm (BaseChatModel): The language model used for generating the new reflection.
        examples (str): Example inputs for the prompt template.
        question (str): The question being addressed.
        scratchpad (str): The scratchpad content related to the question.
        context (Optional[str]): The context of the conversation or query. Defaults to None.
        prompt (str, optional): Prompt template string.
        additional_keys (Dict[str, str]): Additional keys to format the prompt. Defaults to {}

    Returns:
        List[str]: A list with the new reflections.
    """
    reflections = [
        remove_newline(
            _prompt_cot_reflection(
                llm=llm,
                examples=examples,
                question=question,
                scratchpad=scratchpad,
                prompt=prompt,
                additional_keys=additional_keys,
            )
        )
    ]
    return reflections


def cot_reflect(
    reflect_strategy: str,
    llm: BaseChatModel,
    reflections: List[str],
    examples: str,
    question: str,
    scratchpad: str,
    prompt: str,
    additional_keys: Dict[str, str] = {},
) -> List[str]:
    """Performs reflection based on a specified strategy using provided context, question, and scratchpad.

    Used with ReflexionCoT. This function orchestrates different types of reflections based on the strategy provided. It either reflects on the
    last attempt, generates a new reflexion, or combines both approaches. Depending on the strategy, it either uses
    the existing reflections, modifies them, or generates new reflections using the provided language model.

    Args:
        reflect_strategy (str): The reflection strategy to be used ('last_attempt', 'reflexion', or 'last_attempt_and_reflexion').
        llm (BaseChatModel): The language model used for generating new reflections.
        reflections (List[str]): A list of existing reflections.
        examples (str): Example inputs for the prompt template.
        question (str): The question being addressed.
        scratchpad (str): The scratchpad content related to the question.
        prompt (str): Prompt template string.
        additional_keys (Dict[str, str], optional): Additional keys to be passed to the prompt template. Defaults to {}

    Returns:
        List[str]: A list of reflections.

    Raises:
        NotImplementedError: If an unknown reflection strategy is specified.

    Strategy-Specific Parameters:
        - "last_attempt": This strategy uses only 'question' and 'scratchpad'. The 'reflections' list is updated with the current scratchpad.
        - "reflexion": This strategy uses all the parameters. It adds a new reflexion generated by the language model to the 'reflections' list.
        - "last_attempt_and_reflexion": This strategy combines the 'last_attempt' and 'reflexion' strategies.
          It first formats the last attempt using 'question' and 'scratchpad', then adds a new reflexion using all the parameters.
    """
    if reflect_strategy == "last_attempt":
        reflections = cot_reflect_last_attempt(scratchpad)
    elif reflect_strategy == "reflexion":
        reflections = cot_reflect_reflexion(
            llm=llm,
            reflections=reflections,
            examples=examples,
            question=question,
            scratchpad=scratchpad,
            prompt=prompt,
            additional_keys=additional_keys,
        )
    elif reflect_strategy == "last_attempt_and_reflexion":
        reflections = cot_reflect_last_attempt_and_reflexion(
            llm=llm,
            examples=examples,
            question=question,
            scratchpad=scratchpad,
            prompt=prompt,
            additional_keys=additional_keys,
        )
    else:
        raise NotImplementedError(f"Unknown reflection strategy: {reflect_strategy}.")

    return reflections


def _build_react_agent_prompt(
    question: str,
    examples: str,
    reflections: str,
    scratchpad: str,
    max_steps: int,
    prompt: str,
    additional_keys: Dict[str, str] = {},
) -> str:
    """Constructs a ReflexionReAct prompt template for the agent.

    Args:
        question (str): The question being addressed.
        examples (str): Example inputs for the prompt template.
        reflections (List[str]): Existing list of reflections.
        scratchpad (str): The scratchpad content related to the question.
        max_steps (int): Maximum number of steps.
        prompt (str, optional): Prompt template string.
        additional_keys (Dict[str, str]): Additional keys to format the prompt. Defaults to {}.

    Returns:
        str: A formatted prompt template ready for use.
    """
    prompt = PromptTemplate(
        input_variables=["examples", "reflections", "question", "scratchpad"]
        + list(additional_keys.keys()),
        template=prompt,
    ).format(
        question=question,
        examples=examples,
        reflections=reflections,
        scratchpad=scratchpad,
        max_steps=max_steps,
        **additional_keys,
    )

    return prompt


def _prompt_react_agent(
    llm: BaseChatModel,
    question: str,
    examples: str,
    reflections: str,
    scratchpad: str,
    max_steps: int,
    prompt: str,
    additional_keys: Dict[str, str] = {},
) -> str:
    """Generates a ReAct prompt for thought and action.

    Used with ReflexionReAct.

    Args:
        llm (BaseChatModel): The language model to be used for generating the reflection.
        question (str): The question being addressed.
        examples (str): Example inputs for the prompt template.
        reflections (List[str]): Existing list of reflections.
        scratchpad (str): The scratchpad content related to the question.
        max_steps (int): Maximum number of steps.
        prompt (str, optional): Prompt template string.
        additional_keys (Dict[str, str]): Additional keys to format the prompt. Defaults to {}.

    Returns:
        str: The generated reflection prompt.
    """
    prompt = _build_react_agent_prompt(
        question=question,
        examples=examples,
        reflections=reflections,
        scratchpad=scratchpad,
        max_steps=max_steps,
        prompt=prompt,
        additional_keys=additional_keys,
    )
    print(
        "<PROMPT AGENT===============================================================>"
    )
    print(prompt)
    print(
        "<PROMPT AGENT===============================================================>"
    )
    out = llm(
        [
            HumanMessage(
                content=prompt,
            )
        ]
    ).content
    print("<OUT AGENT===============================================================>")
    print(repr(out))
    print("<OUT AGENT===============================================================>")
    assert isinstance(out, str)
    return out


def _is_halted(
    finished: bool,
    step_idx: int,
    question: str,
    scratchpad: str,
    examples: str,
    reflections: str,
    max_steps: int,
    max_tokens: int,
    enc: Encoding,
    prompt: str,
    additional_keys: Dict[str, str] = {},
) -> bool:
    """Determines whether the agent's operation should be halted.

    This function checks if the operation should be halted based on three conditions:
    completion (finished), exceeding maximum steps, or exceeding maximum token limit.
    The token limit is evaluated based on the encoded length of the prompt.

    Args:
        finished (bool): Flag indicating if the operation is completed.
        step_idx (int): Current step number.
        question (str): The question being processed.
        scratchpad (str): The scratchpad content.
        examples (str): Fewshot examples.
        reflections (str): Reflections.
        max_steps (int): Maximum allowed steps.
        max_tokens (int): Maximum allowed token count.
        enc (Encoding): The encoder to calculate token length.
        prompt (str): Prompt template string.
        additional_keys (Dict[str, str]): Additional keys to format the prompt. Defaults to

    Returns:
        bool: True if the operation should be halted, False otherwise.
    """
    over_max_steps = step_idx > max_steps
    over_token_limit = (
        len(
            enc.encode(
                _build_react_agent_prompt(
                    examples=examples,
                    reflections=reflections,
                    question=question,
                    scratchpad=scratchpad,
                    max_steps=max_steps,
                    prompt=prompt,
                    additional_keys=additional_keys,
                )
            )
        )
        > max_tokens
    )
    return finished or over_max_steps or over_token_limit


def _build_react_reflection_prompt(
    question: str,
    examples: str,
    scratchpad: str,
    prompt: str,
    additional_keys: Dict[str, str] = {},
) -> str:
    """Constructs a ReflexionReAct prompt template for reflection.

    Args:
        question (str): The question being addressed.
        examples (str): Example inputs for the prompt template.
        scratchpad (str): The scratchpad content related to the question.
        prompt (str, optional): Reflect prompt template string.
        additional_keys (Dict[str, str]): Additional keys to format the prompt. Defaults to {}.

    Returns:
        str: A formatted prompt template ready for use.
    """
    prompt = PromptTemplate(
        input_variables=["examples", "question", "scratchpad"],
        template=prompt,
    ).format(
        question=question,
        examples=examples,
        scratchpad=scratchpad,
        **additional_keys,
    )

    return prompt


def _prompt_react_reflection(
    llm: BaseChatModel,
    question: str,
    examples: str,
    scratchpad: str,
    prompt: str,
    additional_keys: Dict[str, str] = {},
) -> str:
    """Generates a reflection prompt.

    Used with ReflexionReAct.

    Args:
        llm (BaseChatModel): The language model to be used for generating the reflection.
        question (str): The question being addressed.
        examples (str): Example inputs for the prompt template.
        scratchpad (str): The scratchpad content related to the question.
        prompt (str, optional): Reflect prompt template string.
        additional_keys (Dict[str, str]): Additional keys to format the prompt. Defaults to {}.

    Returns:
        str: The generated reflection prompt.
    """
    prompt = _build_react_reflection_prompt(
        question=question,
        examples=examples,
        scratchpad=scratchpad,
        prompt=prompt,
        additional_keys=additional_keys,
    )
    print(
        "<PROMPT REFLECT===============================================================>"
    )
    print(prompt)
    print(
        "<PROMPT REFLECT===============================================================>"
    )
    out = llm(
        [
            HumanMessage(
                content=prompt,
            )
        ]
    ).content
    print(
        "<OUT REFLECT===============================================================>"
    )
    print(repr(out))
    print(
        "<OUT REFLECT===============================================================>"
    )
    assert isinstance(out, str)
    return out


def react_reflect_last_attempt(scratchpad: str) -> List[str]:
    """Performs a reflection based on the last attempt (scratchpad).

    Used with ReflexionReAct.

    Args:
        question (str): The question associated with the last attempt.
        scratchpad (str): The scratchpad content from the last attempt.

    Returns:
        List[str]: A list with the scratchpad content.
    """
    return [scratchpad]


def react_reflect_reflexion(
    llm: BaseChatModel,
    reflections: List[str],
    question: str,
    examples: str,
    scratchpad: str,
    prompt: str,
    additional_keys: Dict[str, str] = {},
) -> List[str]:
    """Perform reflexion-based reflecting.

    Used with ReflexionReAct. This function uses a language model to generate a new reflection based on the provided context, question,
    and scratchpad. The new reflection is added to the existing list of reflections.

    Args:
        llm (BaseChatModel): The language model used for generating the reflection.
        reflections (List[str]): Existing list of reflections.
        question (str): The question being addressed.
        examples (str): Example inputs for the prompt template.
        scratchpad (str): The scratchpad content related to the question.
        prompt (str, optional): Reflect prompt template string.
        additional_keys (Dict[str, str]): Additional keys to format the prompt. Defaults to {}.

    Returns:
        List[str]: An updated list of reflections.
    """
    new_reflection = remove_newline(
        _prompt_react_reflection(
            llm=llm,
            question=question,
            examples=examples,
            scratchpad=scratchpad,
            prompt=prompt,
            additional_keys=additional_keys,
        )
    )
    reflections += [new_reflection]
    return reflections


def react_reflect_last_attempt_and_reflexion(
    llm: BaseChatModel,
    question: str,
    examples: str,
    scratchpad: str,
    prompt: str,
    additional_keys: Dict[str, str] = {},
) -> List[str]:
    """Performs reflection with the reflection of the last attempt and reflexion.

    Used with ReflexionReAct.

    Args:
        llm (BaseChatModel): The language model used for generating the new reflection.
        question (str): The question being addressed.
        examples (str): Example inputs for the prompt template.
        scratchpad (str): The scratchpad content related to the question.
        prompt (str, optional): Reflect prompt template string.
        additional_keys (Dict[str, str]): Additional keys to format the prompt. Defaults to {}.

    Returns:
        List[str]: A list with the new reflections.
    """
    reflections = [
        remove_newline(
            _prompt_react_reflection(
                llm=llm,
                question=question,
                examples=examples,
                scratchpad=scratchpad,
                prompt=prompt,
                additional_keys=additional_keys,
            )
        )
    ]
    return reflections


def react_reflect(
    reflect_strategy: str,
    llm: BaseChatModel,
    reflections: List[str],
    question: str,
    examples: str,
    scratchpad: str,
    prompt: str,
    additional_keys: Dict[str, str] = {},
) -> List[str]:
    """Performs reflection based on a specified strategy using the provided question and scratchpad.

    Used with ReflexionReAct. This function orchestrates different types of reflections based on the strategy provided. It either reflects on the
    last attempt, generates a new reflexion, or combines both approaches. Depending on the strategy, it either uses
    the existing reflections, modifies them, or generates new reflections using the provided language model.

    Args:
        reflect_strategy (str): The reflection strategy to be used ('last_attempt', 'reflexion', or 'last_attempt_and_reflexion').
        llm (BaseChatModel): The language model used for generating new reflections.
        reflections (List[str]): A list of existing reflections.
        question (str): The question being addressed.
        examples (str): Example inputs for the prompt template.
        scratchpad (str): The scratchpad content related to the question.
        prompt (str, optional): Reflect prompt template string.
        additional_keys (Dict[str, str]): Additional keys to format the prompt. Defaults to {}.

    Returns:
        List[str]: A tuple containing the updated list of reflections.

    Raises:
        NotImplementedError: If an unknown reflection strategy is specified.

    Strategy-Specific Parameters:
        - "last_attempt": This strategy uses only 'question' and 'scratchpad'. The 'reflections' list is updated with the current scratchpad.
        - "reflexion": This strategy uses all the parameters. It adds a new reflexion generated by the language model to the 'reflections' list.
        - "last_attempt_and_reflexion": This strategy combines the 'last_attempt' and 'reflexion' strategies.
          It first formats the last attempt using 'question' and 'scratchpad', then adds a new reflexion using all the parameters.
    """
    if reflect_strategy == "last_attempt":
        reflections = react_reflect_last_attempt(scratchpad)
    elif reflect_strategy == "reflexion":
        reflections = react_reflect_reflexion(
            llm=llm,
            reflections=reflections,
            question=question,
            examples=examples,
            scratchpad=scratchpad,
            prompt=prompt,
            additional_keys=additional_keys,
        )
    elif reflect_strategy == "last_attempt_and_reflexion":
        reflections = react_reflect_last_attempt_and_reflexion(
            llm=llm,
            question=question,
            examples=examples,
            scratchpad=scratchpad,
            prompt=prompt,
            additional_keys=additional_keys,
        )
    else:
        raise NotImplementedError(f"Unknown reflection strategy: {reflect_strategy}.")

    return reflections
