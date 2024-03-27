"""Unit tests for ReAct functional module."""
import tiktoken

from langchain_community.chat_models.fake import FakeListChatModel

from discussion_agents.cog.functional.react import (
    _build_agent_prompt,
    _is_halted,
    _prompt_agent,
)
from discussion_agents.cog.prompts.react import REACT_WEBTHINK_SIMPLE6_FEWSHOT_EXAMPLES


def test__build_agent_prompt() -> None:
    """Test _build_agent_prompt function."""
    prompt = _build_agent_prompt(
        question="",
        scratchpad="",
        examples=REACT_WEBTHINK_SIMPLE6_FEWSHOT_EXAMPLES,
        max_steps=1,
    )

    gt_out = (
        "Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types: \n"
        "(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.\n"
        "(2) Lookup[keyword], which returns the next sentence containing keyword in the last passage successfully found by Search.\n"
        "(3) Finish[answer], which returns the answer and finishes the task.\n"
        "You have a maximum of 1 steps.\n\n"
        "Here are some examples:\n"
        "Question: What is the elevation range for the area that the eastern sector of the Colorado orogeny extends into?\n"
        "Thought 1: I need to search Colorado orogeny, find the area that the eastern sector of the Colorado orogeny extends into, then find the elevation range of the area.\n"
        "Action 1: Search[Colorado orogeny]\n"
        "Observation 1: The Colorado orogeny was an episode of mountain building (an orogeny) in Colorado and surrounding areas.\n"
        "Thought 2: It does not mention the eastern sector. So I need to look up eastern sector.\n"
        "Action 2: Lookup[eastern sector]\n"
        "Observation 2: (Result 1 / 1) The eastern sector extends into the High Plains and is called the Central Plains orogeny.\n"
        "Thought 3: The eastern sector of Colorado orogeny extends into the High Plains. So I need to search High Plains and find its elevation range.\n"
        "Action 3: Search[High Plains]\n"
        "Observation 3: High Plains refers to one of two distinct land regions:\n"
        "Thought 4: I need to instead search High Plains (United States).\n"
        "Action 4: Search[High Plains (United States)]\n"
        "Observation 4: The High Plains are a subregion of the Great Plains. From east to west, the High Plains rise in elevation from around 1,800 to 7,000 ft (550 to 2,130 m).[3]\n"
        "Thought 5: High Plains rise in elevation from around 1,800 to 7,000 ft, so the answer is 1,800 to 7,000 ft.\n"
        "Action 5: Finish[1,800 to 7,000 ft]\n"
        'Question: Musician and satirist Allie Goertz wrote a song about the "The Simpsons" character Milhouse, who Matt Groening named after who?\n'
        'Thought 1: The question simplifies to "The Simpsons" character Milhouse is named after who. I only need to search Milhouse and find who it is named after.\n'
        "Action 1: Search[Milhouse]\n"
        "Observation 1: Milhouse Mussolini Van Houten is a recurring character in the Fox animated television series The Simpsons voiced by Pamela Hayden and created by Matt Groening.\n"
        'Thought 2: The paragraph does not tell who Milhouse is named after, maybe I can look up "named after".\n'
        "Action 2: Lookup[named after]\n"
        "Observation 2: (Result 1 / 1) Milhouse was named after U.S. president Richard Nixon, whose middle name was Milhous. \n"
        "Thought 3: Milhouse was named after U.S. president Richard Nixon, so the answer is Richard Nixon.\n"
        "Action 3: Finish[Richard Nixon]\n"
        "Question: Which documentary is about Finnish rock groups, Adam Clayton Powell or The Saimaa Gesture?\n"
        "Thought 1: I need to search Adam Clayton Powell and The Saimaa Gesture, and find which documentary is about Finnish rock groups.\n"
        "Action 1: Search[Adam Clayton Powell]\n"
        "Observation 1: Could not find [Adam Clayton Powell]. Similar: ['Adam Clayton Powell III', 'Seventh Avenue (Manhattan)', 'Adam Clayton Powell Jr. State Office Building', 'Isabel Washington Powell', 'Adam Powell', 'Adam Clayton Powell (film)', 'Giancarlo Esposito'].\n"
        "Thought 2: To find the documentary, I can search Adam Clayton Powell (film).\n"
        "Action 2: Search[Adam Clayton Powell (film)]\n"
        "Observation 2: Adam Clayton Powell is a 1989 American documentary film directed by Richard Kilberg.\n"
        "The film is about the rise and fall of influential African-American politician Adam Clayton Powell Jr.[3][4] It was later aired as part of the PBS series The American Experience.\n"
        "Thought 3: Adam Clayton Powell (film) is a documentary about an African-American politician, not Finnish rock groups. So the documentary about Finnish rock groups must instead be The Saimaa Gesture.\n"
        "Action 3: Finish[The Saimaa Gesture]\n"
        "Question: What profession does Nicholas Ray and Elia Kazan have in common?\n"
        "Thought 1: I need to search Nicholas Ray and Elia Kazan, find their professions, then find the profession they have in common.\n"
        "Action 1: Search[Nicholas Ray]\n"
        "Observation 1: Nicholas Ray (born Raymond Nicholas Kienzle Jr., August 7, 1911 – June 16, 1979) was an American film director, screenwriter, and actor best known for the 1955 film Rebel Without a Cause.\n"
        "Thought 2: Professions of Nicholas Ray are director, screenwriter, and actor. I need to search Elia Kazan next and find his professions.\n"
        "Action 2: Search[Elia Kazan]\n"
        "Observation 2: Elia Kazan was an American film and theatre director, producer, screenwriter and actor.\n"
        "Thought 3: Professions of Elia Kazan are director, producer, screenwriter, and actor. So profession Nicholas Ray and Elia Kazan have in common is director, screenwriter, and actor.\n"
        "Action 3: Finish[director, screenwriter, actor]\n"
        "Question: Which magazine was started first Arthur's Magazine or First for Women?\n"
        "Thought 1: I need to search Arthur's Magazine and First for Women, and find which was started first.\n"
        "Action 1: Search[Arthur's Magazine]\n"
        "Observation 1: Arthur's Magazine (1844-\x80\x931846) was an American literary periodical published in Philadelphia in the 19th century. \n"
        "Thought 2: Arthur's Magazine was started in 1844. I need to search First for Women next.\n"
        "Action 2: Search[First for Women]\n"
        "Observation 2: First for Women is a woman's magazine published by Bauer Media Group in the USA.[1] The magazine was started in 1989. \n"
        "Thought 3: First for Women was started in 1989. 1844 (Arthur's Magazine) < 1989 (First for Women), so Arthur's Magazine was started first.\n"
        "Action 3: Finish[Arthur's Magazine]\n"
        "Question: Were Pavel Urysohn and Leonid Levin known for the same type of work?\n"
        "Thought 1: I need to search Pavel Urysohn and Leonid Levin, find their types of work, then find if they are the same.\n"
        "Action 1: Search[Pavel Urysohn]\n"
        "Observation 1: Pavel Samuilovich Urysohn (February 3, 1898 â\x80\x93 August 17, 1924) was a Soviet mathematician who is best known for his contributions in dimension theory.\n"
        "Thought 2: Pavel Urysohn is a mathematician. I need to search Leonid Levin next and find its type of work.\n"
        "Action 2: Search[Leonid Levin]\n"
        "Observation 2: Leonid Anatolievich Levin is a Soviet-American mathematician and computer scientist. \n"
        "Thought 3: Leonid Levin is a mathematician and computer scientist. So Pavel Urysohn and Leonid Levin have the same type of work. \n"
        "Action 3: Finish[yes]\n"
        "(END OF EXAMPLES)\n\n"
        "Question: "
    )

    assert isinstance(prompt, str)
    assert prompt == gt_out

    gt_out = "  examples 1"
    out = _build_agent_prompt(
        question="",
        scratchpad="",
        examples="examples",
        max_steps=1,
        prompt="{question} {scratchpad} {examples} {max_steps}",
    )
    assert out == gt_out


def test__prompt_agent() -> None:
    """Test _prompt_agent function."""
    out = _prompt_agent(
        llm=FakeListChatModel(responses=["1"]),
        question="",
        scratchpad="",
        examples=REACT_WEBTHINK_SIMPLE6_FEWSHOT_EXAMPLES,
        max_steps=1,
    )
    assert isinstance(out, str)
    assert out == "1"

    # Test with custom prompt template string.
    out = _prompt_agent(
        llm=FakeListChatModel(responses=["1"]),
        question="",
        scratchpad="",
        examples=REACT_WEBTHINK_SIMPLE6_FEWSHOT_EXAMPLES,
        max_steps=1,
        prompt="{question} {scratchpad} {examples} {max_steps}",
    )
    assert isinstance(out, str)
    assert out == "1"


def test__is_halted() -> None:
    """Test _is_halted function."""
    gpt3_5_turbo_enc = tiktoken.encoding_for_model("gpt-3.5-turbo")

    # Test when finish is true.
    assert _is_halted(
        True,
        1,
        "question",
        "scratchpad",
        REACT_WEBTHINK_SIMPLE6_FEWSHOT_EXAMPLES,
        10,
        100,
        gpt3_5_turbo_enc,
    )

    # Test when step_n exceeds max_steps.
    assert _is_halted(
        False,
        11,
        "question",
        "scratchpad",
        REACT_WEBTHINK_SIMPLE6_FEWSHOT_EXAMPLES,
        10,
        100,
        gpt3_5_turbo_enc,
    )

    # Test when encoded prompt exceeds max_tokens.
    assert _is_halted(
        False,
        1,
        "question",
        "scratchpad",
        REACT_WEBTHINK_SIMPLE6_FEWSHOT_EXAMPLES,
        10,
        10,
        gpt3_5_turbo_enc,
    )

    # Test when none of the conditions for halting are met.
    assert not _is_halted(
        False,
        1,
        "question",
        "scratchpad",
        REACT_WEBTHINK_SIMPLE6_FEWSHOT_EXAMPLES,
        10,
        100000,
        gpt3_5_turbo_enc,
    )

    # Test edge case when step_n equals max_steps.
    assert _is_halted(
        False,
        10,
        "question",
        "scratchpad",
        REACT_WEBTHINK_SIMPLE6_FEWSHOT_EXAMPLES,
        10,
        100,
        gpt3_5_turbo_enc,
    )

    # Test edge case when encoded prompt equals max_tokens.
    assert _is_halted(
        False,
        1,
        "question",
        "scratchpad",
        REACT_WEBTHINK_SIMPLE6_FEWSHOT_EXAMPLES,
        10,
        1603,
        gpt3_5_turbo_enc,
    )

    # Test with custom prompt template string.
    assert not _is_halted(
        False,
        1,
        "question",
        "scratchpad",
        REACT_WEBTHINK_SIMPLE6_FEWSHOT_EXAMPLES,
        10,
        1603,
        gpt3_5_turbo_enc,
        "{question} {scratchpad} {examples} {max_steps}",
    )
