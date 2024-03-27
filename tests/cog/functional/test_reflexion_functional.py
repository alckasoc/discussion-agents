"""Unit tests for Reflexion functional methods."""
import pytest
import tiktoken

from langchain_community.chat_models.fake import FakeListChatModel

from discussion_agents.cog.functional.reflexion import (
    _build_cot_agent_prompt,
    _build_cot_reflection_prompt,
    _build_react_agent_prompt,
    _build_react_reflection_prompt,
    _format_last_attempt,
    _format_reflections,
    _is_halted,
    _prompt_cot_agent,
    _prompt_cot_reflection,
    _prompt_react_agent,
    _prompt_react_reflection,
    _truncate_scratchpad,
    cot_reflect,
    cot_reflect_last_attempt,
    cot_reflect_last_attempt_and_reflexion,
    cot_reflect_reflexion,
    react_reflect,
    react_reflect_last_attempt,
    react_reflect_last_attempt_and_reflexion,
    react_reflect_reflexion,
)
from discussion_agents.cog.prompts.react import REACT_WEBTHINK_SIMPLE6_FEWSHOT_EXAMPLES
from discussion_agents.cog.prompts.reflexion import (
    REFLEXION_COT_FEWSHOT_EXAMPLES,
    REFLEXION_COT_FEWSHOT_EXAMPLES_NO_CONTEXT,
    REFLEXION_COT_REFLECT_FEWSHOT_EXAMPLES,
    REFLEXION_COT_REFLECT_FEWSHOT_EXAMPLES_NO_CONTEXT,
    REFLEXION_REACT_REFLECT_FEWSHOT_EXAMPLES,
)


def test__truncate_scratchpad() -> None:
    """Test _truncate_scratchpad function."""
    # Test non-truncated case.
    scratchpad = "Observation: This is a test.\nAnother line."
    truncated = _truncate_scratchpad(scratchpad, 1600)
    assert truncated == scratchpad

    # Test truncated case.
    gt_out = "Observation: [truncated wikipedia excerpt]\nAnother line."
    long_observation = "Observation: " + "long text " * 100
    scratchpad = long_observation + "\nAnother line."
    truncated = _truncate_scratchpad(scratchpad, 100)
    assert long_observation not in truncated
    assert "truncated wikipedia excerpt" in truncated
    assert gt_out == truncated

    # Test non-truncated case with random format.
    scratchpad = "Regular line 1.\nRegular line 2."
    truncated = _truncate_scratchpad(scratchpad, 1600)
    assert truncated == scratchpad

    # Test truncated case with long text and multiple observations.
    gt_out = "Observation: short text\nObservation: [truncated wikipedia excerpt]"
    observation1 = "Observation: short text"
    observation2 = "Observation: " + "long text " * 100
    scratchpad = observation1 + "\n" + observation2
    truncated = _truncate_scratchpad(scratchpad, 100)
    assert observation1 in truncated, "Shorter observation should remain"
    assert observation2 not in truncated, "Longer observation should be truncated"
    assert gt_out == truncated


def test__format_reflections() -> None:
    """Test _format_reflections function."""
    # Test empty.
    reflections = []
    assert _format_reflections(reflections) == ""

    # Test non-empty reflections.
    reflections = ["Reflection 1", "Reflection 2"]
    expected_result = "You have attempted to answer following question before and failed. The following reflection(s) give a plan to avoid failing to answer the question in the same way you did previously. Use them to improve your strategy of correctly answering the given question.\nReflections:\n- Reflection 1\n- Reflection 2"
    assert _format_reflections(reflections) == expected_result

    # Test reflections with spaces.
    reflections = ["  Reflection 1  ", "  Reflection 2"]
    expected_result = "You have attempted to answer following question before and failed. The following reflection(s) give a plan to avoid failing to answer the question in the same way you did previously. Use them to improve your strategy of correctly answering the given question.\nReflections:\n- Reflection 1\n- Reflection 2"
    assert _format_reflections(reflections) == expected_result

    # Test custom header.
    reflections = ["Reflection"]
    custom_header = "Custom Header: "
    expected_result = "Custom Header: Reflections:\n- Reflection"
    assert _format_reflections(reflections, custom_header) == expected_result


def test__format_last_attempt() -> None:
    """Test _format_last_attempt function."""
    # Test simple case.
    question = "What is the capital of France?"
    scratchpad = "The capital of France is Paris."
    expected_format = "You have attempted to answer the following question before and failed. Below is the last trial you attempted to answer the question.\nQuestion: What is the capital of France?\nThe capital of France is Paris.\n(END PREVIOUS TRIAL)\n"
    result = _format_last_attempt(question, scratchpad)
    assert result == expected_format


def test__build_cot_agent_prompt() -> None:
    """Test _build_cot_agent_prompt function."""
    gt_out = 'Solve a question answering task by having a Thought, then Finish with your answer. Thought can reason about the current situation. Finish[answer] returns the answer and finishes the task.\nHere are some examples:\n\n(END OF EXAMPLES)\n\n\n\nQuestion: '
    out = _build_cot_agent_prompt(
        examples="", reflections="", question="", scratchpad="", context=""
    )
    assert out == gt_out

    gt_out = 'Solve a question answering task by having a Thought, then Finish with your answer. Thought can reason about the current situation. Finish[answer] returns the answer and finishes the task.\nHere are some examples:\n\n(END OF EXAMPLES)\n\n\n\nQuestion: '
    out = _build_cot_agent_prompt(
        examples="", reflections="", question="", scratchpad="", context=None
    )
    assert out == gt_out


def test__prompt_cot_agent() -> None:
    """Test _prompt_cot_agent function."""
    q = "VIVA Media AG changed it's name in 2004. What does their new acronym stand for?"
    context = 'VIVA Media GmbH (until 2004 "VIVA Media AG") is a music television network originating from Germany. It was founded for broadcast of VIVA Germany as VIVA Media AG in 1993 and has been owned by their original concurrent Viacom, the parent company of MTV, since 2004. Viva channels exist in some European countries; the first spin-offs were launched in Poland and Switzerland in 2000.\n\nA Gesellschaft mit beschränkter Haftung (] , abbreviated GmbH ] and also GesmbH in Austria) is a type of legal entity very common in Germany, Austria, Switzerland (where it is equivalent to a S.à r.l.) and Liechtenstein. In the United States, the equivalent type of entity is the limited liability company (LLC). The name of the GmbH form emphasizes the fact that the owners ("Gesellschafter", also known as members) of the entity are not personally liable for the company\'s debts. "GmbH"s are considered legal persons under German and Austrian law. Other variations include mbH (used when the term "Gesellschaft" is part of the company name itself), and gGmbH ("gemeinnützige" GmbH) for non-profit companies.'

    # Test with context.
    out = _prompt_cot_agent(
        llm=FakeListChatModel(responses=["1"]),
        examples="",
        reflections="",
        question="",
        scratchpad="",
        context="",
    )
    assert isinstance(out, str)
    assert out == "1"

    # Test with no context.
    out = _prompt_cot_agent(
        llm=FakeListChatModel(responses=["1"]),
        examples="",
        reflections="",
        question="",
        scratchpad="",
        context=None,
    )
    assert isinstance(out, str)
    assert out == "1"

    # Test simple case (no reflection) with context.
    gt_out = (
        "Let's think step by step. VIVA Media AG changed its name to VIVA Media GmbH in 2004. "
        'GmbH stands for "Gesellschaft mit beschränkter Haftung" which translates to "company with '
        'limited liability" in English.Action: Finish[company with limited liability]'
    )
    responses = [
        (
            "Let's think step by step. VIVA Media AG changed its name to VIVA Media GmbH in 2004. "
            'GmbH stands for "Gesellschaft mit beschränkter Haftung" which translates to "company with limited liability" '
            "in English.\nAction: Finish[company with limited liability]"
        )
    ]
    out = _prompt_cot_agent(
        llm=FakeListChatModel(responses=responses),
        examples=REFLEXION_COT_FEWSHOT_EXAMPLES,
        reflections="",
        question=q,
        scratchpad="\nThought:",
        context=context,
    )
    assert out == gt_out

    # Test simple case (no reflection) with no context.
    gt_out = (
        "Thought: Let's think step by step. The new acronym for VIVA Media AG after changing its name in "
        '2004 is "Vivendi Visual and Interactive." Action: Finish[Vivendi Visual and Interactive]'
    )
    responses = [
        (
            "Thought: Let's think step by step. The new acronym for VIVA Media AG after changing its name in 2004 "
            'is "Vivendi Visual and Interactive." \nAction: Finish[Vivendi Visual and Interactive]'
        )
    ]
    out = _prompt_cot_agent(
        llm=FakeListChatModel(responses=responses),
        examples=REFLEXION_COT_FEWSHOT_EXAMPLES_NO_CONTEXT,
        reflections="",
        question=q,
        scratchpad="\nThought:",
        context=None,
    )
    assert out == gt_out

    # Test simple case (reflection) with context.
    reflections = (
        "You have attempted to answer the following question before and failed. "
        "Below is the last trial you attempted to answer the question.\n"
        "Question: VIVA Media AG changed it's name in 2004. What does their new acronym stand for?\n"
        "Thought: The context provided mentions that VIVA Media AG changed its name in 2004 and has been "
        "owned by Viacom since then. Based on this information, the new acronym for VIVA Media AG is "
        "likely related to Viacom, the parent company of MTV.Action: Finish[Viacom]\n"
        "Action: Finish[Viacom]\nObservation: Answer is INCORRECT\n(END PREVIOUS TRIAL)\n"
    )
    scratchpad = (
        "\nThought: The context provided mentions that VIVA Media AG changed its name in 2004 and has "
        "been owned by Viacom since then. Based on this information, the new acronym for VIVA Media AG is "
        "likely related to Viacom, the parent company of MTV.Action: Finish[Viacom]\n"
        "Action: Finish[Viacom]\nObservation: Answer is INCORRECT\nThought:"
    )
    responses = [
        (
            "The context provided states that VIVA Media AG changed its name to VIVA Media GmbH in 2004. "
            'Since "GmbH" stands for "Gesellschaft mit beschränkter Haftung" in German, the new acronym '
            'for VIVA Media AG is likely VIVA Media GmbH, with "GmbH" representing the legal form of the '
            "company.\nAction: Finish[VIVA Media GmbH]"
        )
    ]
    gt_out = (
        "The context provided states that VIVA Media AG changed its name to VIVA Media GmbH in 2004. "
        'Since "GmbH" stands for "Gesellschaft mit beschränkter Haftung" in German, the new acronym for '
        'VIVA Media AG is likely VIVA Media GmbH, with "GmbH" representing the legal form of the company.'
        "Action: Finish[VIVA Media GmbH]"
    )
    out = _prompt_cot_agent(
        llm=FakeListChatModel(responses=responses),
        examples=REFLEXION_COT_FEWSHOT_EXAMPLES,
        reflections=reflections,
        question=q,
        scratchpad=scratchpad,
        context=context,
    )
    assert out == gt_out

    # Test simple case (reflection) with no context.
    reflections = (
        "You have attempted to answer the following question before and failed. Below is the last trial "
        "you attempted to answer the question.\nQuestion: VIVA Media AG changed it's name in 2004. "
        "What does their new acronym stand for?\nThought: Let's think step by step. VIVA Media AG "
        'changed its name to Constantin Film AG in 2004. The new acronym stands for "Constantin Film."'
        "Action: Finish[Constantin Film]\nAction: Finish[Constantin Film]\n"
        "Observation: Answer is INCORRECT\n(END PREVIOUS TRIAL)\n"
    )
    scratchpad = (
        "\nThought: Let's think step by step. VIVA Media AG changed its name to Constantin Film AG in 2004. "
        'The new acronym stands for "Constantin Film."Action: Finish[Constantin Film]\n'
        "Action: Finish[Constantin Film]\nObservation: Answer is INCORRECT\nThought:"
    )
    responses = [
        (
            "I made a mistake in my previous attempts. Let's think more carefully this time. "
            'The acronym "AG" stands for "Aktiengesellschaft" in German, which translates to '
            '"stock corporation" in English. So the new acronym for VIVA Media AG after the name '
            'change is "Constantin Film Aktiengesellschaft."\nFinish[Constantin Film Aktiengesellschaft]'
        )
    ]
    gt_out = (
        'I made a mistake in my previous attempts. Let\'s think more carefully this time. The acronym "AG" '
        'stands for "Aktiengesellschaft" in German, which translates to "stock corporation" in English. '
        'So the new acronym for VIVA Media AG after the name change is "Constantin Film Aktiengesellschaft."'
        "Finish[Constantin Film Aktiengesellschaft]"
    )
    out = _prompt_cot_agent(
        llm=FakeListChatModel(responses=responses),
        examples=REFLEXION_COT_FEWSHOT_EXAMPLES_NO_CONTEXT,
        reflections=reflections,
        question=q,
        scratchpad=scratchpad,
        context=None,
    )
    assert out == gt_out


def test__build_cot_reflection_prompt() -> None:
    """Test _build_cot_reflection_prompt function."""
    gt_out = 'You are an advanced reasoning agent that can improve based on self refection. You will be given a previous reasoning trial in which you were given a question to answer. You were unsuccessful in answering the question either because you guessed the wrong answer with Finish[<answer>] or there is a phrasing discrepancy with your provided answer and the answer key. In a few sentences, Diagnose a possible reason for failure or phrasing discrepancy and devise a new, concise, high level plan that aims to mitigate the same failure. Use complete sentences.\nHere are some examples:\n\n(END OF EXAMPLES)\n\nPrevious trial:\nQuestion: \n\nReflection:'
    out = _build_cot_reflection_prompt(
        examples="", question="", scratchpad="", context=""
    )
    assert out == gt_out

    gt_out = 'You are an advanced reasoning agent that can improve based on self refection. You will be given a previous reasoning trial in which you were given a question to answer. You were unsuccessful in answering the question either because you guessed the wrong answer with Finish[<answer>] or there is a phrasing discrepancy with your provided answer and the answer key. In a few sentences, Diagnose a possible reason for failure or phrasing discrepancy and devise a new, concise, high level plan that aims to mitigate the same failure. Use complete sentences.\nHere are some examples:\n\n(END OF EXAMPLES)\n\nPrevious trial:\nQuestion: \n\nReflection:'
    out = _build_cot_reflection_prompt(
        examples="", question="", scratchpad="", context=None
    )
    assert out == gt_out

    # Test with custom prompt.
    gt_out = '  '
    out = _build_cot_reflection_prompt(
        examples="",
        question="",
        scratchpad="",
        context=None,
        prompt="{examples} {question} {scratchpad}",
    )
    assert out == gt_out


def test__prompt_cot_reflection() -> None:
    """Test _prompt_cot_reflection function."""
    q = "VIVA Media AG changed it's name in 2004. What does their new acronym stand for?"
    context = 'VIVA Media GmbH (until 2004 "VIVA Media AG") is a music television network originating from Germany. It was founded for broadcast of VIVA Germany as VIVA Media AG in 1993 and has been owned by their original concurrent Viacom, the parent company of MTV, since 2004. Viva channels exist in some European countries; the first spin-offs were launched in Poland and Switzerland in 2000.\n\nA Gesellschaft mit beschränkter Haftung (] , abbreviated GmbH ] and also GesmbH in Austria) is a type of legal entity very common in Germany, Austria, Switzerland (where it is equivalent to a S.à r.l.) and Liechtenstein. In the United States, the equivalent type of entity is the limited liability company (LLC). The name of the GmbH form emphasizes the fact that the owners ("Gesellschafter", also known as members) of the entity are not personally liable for the company\'s debts. "GmbH"s are considered legal persons under German and Austrian law. Other variations include mbH (used when the term "Gesellschaft" is part of the company name itself), and gGmbH ("gemeinnützige" GmbH) for non-profit companies.'

    # Test with context.
    out = _prompt_cot_reflection(
        llm=FakeListChatModel(responses=["1"]),
        examples="",
        question="",
        scratchpad="",
        context="",
    )
    assert isinstance(out, str)
    assert out == "1"

    # Test with no context.
    out = _prompt_cot_reflection(
        llm=FakeListChatModel(responses=["1"]),
        examples="",
        question="",
        scratchpad="",
        context=None,
    )
    assert isinstance(out, str)
    assert out == "1"

    # Test simple case with context.
    scratchpad = (
        "\nThought: The question is asking for the acronym that VIVA Media AG changed to in 2004. "
        "Based on the context provided, I know that VIVA Media AG was renamed to VIVA Media GmbH in 2004. "
        "Action: Finish[VIVA Media GmbH]\nAction: Finish[VIVA Media GmbH]\nObservation: Answer is INCORRECT"
    )
    responses = [
        (
            "The reason for the failure in answering the question could be that the provided answer "
            '"Company with Limited Liability" does not exactly match the full German translation '
            '"Gesellschaft mit beschränkter Haftung" which stands for "company with limited liability" '
            "in English. To mitigate this issue in the future, a more concise and accurate response could "
            'be simply "Limited Liability Company" to align closely with the German term. This will ensure '
            "a more precise match with the expected answer key."
        )
    ]
    gt_out = (
        "The reason for the failure in answering the question could be that the provided answer "
        '"Company with Limited Liability" does not exactly match the full German translation '
        '"Gesellschaft mit beschränkter Haftung" which stands for "company with limited liability" '
        "in English. To mitigate this issue in the future, a more concise and accurate response could "
        'be simply "Limited Liability Company" to align closely with the German term. This will '
        "ensure a more precise match with the expected answer key."
    )
    out = _prompt_cot_reflection(
        llm=FakeListChatModel(responses=responses),
        examples=REFLEXION_COT_REFLECT_FEWSHOT_EXAMPLES,
        question=q,
        scratchpad=scratchpad,
        context=context,
    )
    assert out == gt_out

    # Test simple case with no context.
    scratchpad = (
        "\nThought: Let's think step by step. VIVA Media AG changed its name to VHM Medien in 2004. "
        "VHM Medien stands for Video Home Media.Action: Finish[Video Home Media]\n"
        "Action: Finish[Video Home Media]\nObservation: Answer is INCORRECT"
    )
    responses = [
        (
            "My reasoning for the acronym for VHM Medien being Video Home Media failed because I did "
            "not fully understand the context of the question. In the future, when attempting this "
            "question, I should focus on researching the specific name change of VIVA Media AG in "
            "2004 to avoid confusion. To mitigate this failure, I will double-check the exact name "
            "change and acronym before providing an answer."
        )
    ]
    gt_out = (
        "My reasoning for the acronym for VHM Medien being Video Home Media failed because I did not "
        "fully understand the context of the question. In the future, when attempting this question, "
        "I should focus on researching the specific name change of VIVA Media AG in 2004 to avoid "
        "confusion. To mitigate this failure, I will double-check the exact name change and acronym "
        "before providing an answer."
    )
    out = _prompt_cot_reflection(
        llm=FakeListChatModel(responses=responses),
        examples=REFLEXION_COT_REFLECT_FEWSHOT_EXAMPLES_NO_CONTEXT,
        question=q,
        scratchpad=scratchpad,
        context=None,
    )
    assert out == gt_out


def test_react_reflect_last_attempt() -> None:
    """Test cot_reflect_last_attempt function."""
    scratchpad = ""
    out = cot_reflect_last_attempt(scratchpad)
    assert out == [""]


def test_cot_reflect_reflexion() -> None:
    """Test cot_reflect_reflexion function."""
    out = cot_reflect_reflexion(
        llm=FakeListChatModel(responses=["1"]),
        reflections=[""],
        examples="",
        question="",
        scratchpad="",
        context="",
    )
    assert isinstance(out, list)
    assert out == ["", "1"]


def test_cot_reflect_last_attempt_and_reflexion() -> None:
    """Test cot_reflect_last_attempt_and_reflexion function."""
    out = cot_reflect_last_attempt_and_reflexion(
        llm=FakeListChatModel(responses=["1"]),
        examples="",
        question="",
        scratchpad="",
        context="",
    )
    assert isinstance(out, list)
    assert out == ["1"]


def test_cot_reflect() -> None:
    """Test cot_reflect function."""
    # Invalid strategy.
    with pytest.raises(NotImplementedError):
        out = cot_reflect(
            strategy="invalid input",
            llm=FakeListChatModel(responses=["1"]),
            reflections=[""],
            examples="",
            question="",
            scratchpad="",
            context="",
        )

    # Last attempt.
    out = cot_reflect(
        strategy="last_attempt",
        llm=FakeListChatModel(responses=["1"]),
        reflections=[""],
        examples="",
        question="",
        scratchpad="",
        context="",
    )
    assert out == [""]

    # Reflexion.
    out = cot_reflect(
        strategy="reflexion",
        llm=FakeListChatModel(responses=["1"]),
        reflections=[""],
        examples="",
        question="",
        scratchpad="",
        context="",
    )
    assert isinstance(out, list)
    assert out == ["", "1"]

    # Last attempt and Reflexion.
    out = cot_reflect(
        strategy="last_attempt_and_reflexion",
        llm=FakeListChatModel(responses=["1"]),
        reflections=[""],
        examples="",
        question="",
        scratchpad="",
        context="",
    )
    assert isinstance(out, list)
    assert out == ["1"]


def test__build_react_agent_prompt() -> None:
    """Test _build_react_agent_prompt function."""
    gt_out = "Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types: \n(1) Search[entity], which searches the exact entity on Wikipedia and returns the first paragraph if it exists. If not, it will return some similar entities to search.\n(2) Lookup[keyword], which returns the next sentence containing keyword in the last passage successfully found by Search.\n(3) Finish[answer], which returns the answer and finishes the task.\nYou have a maximum of 1 steps.\n\nHere are some examples:\n\n(END OF EXAMPLES)\n\n\n\nQuestion: "
    out = _build_react_agent_prompt(
        examples="", reflections="", question="", scratchpad="", max_steps=1
    )
    assert out == gt_out


def test__prompt_react_agent() -> None:
    """Test _prompt_react_agent function."""
    q = "VIVA Media AG changed it's name in 2004. What does their new acronym stand for?"

    # Test empty.
    out = _prompt_react_agent(
        llm=FakeListChatModel(responses=["1"]),
        examples="",
        reflections="",
        question="",
        scratchpad="",
        max_steps=1,
    )
    assert isinstance(out, str)
    assert out == "1"

    # Test simple case no reflections.
    responses = [
        (
            "I need to search for VIVA Media AG and find out what their new acronym stands for.\n\nAction: Search[VIVA Media AG]"
        )
    ]
    gt_out = "I need to search for VIVA Media AG and find out what their new acronym stands for.Action: Search[VIVA Media AG]"
    out = _prompt_react_agent(
        llm=FakeListChatModel(responses=responses),
        examples=REACT_WEBTHINK_SIMPLE6_FEWSHOT_EXAMPLES,
        reflections="",
        question=q,
        scratchpad="\nThought:",
        max_steps=1,
    )
    assert out == gt_out

    # Test simple case with reflections.
    responses = [
        (
            "Given the lack of information on VIVA Media AG and their name change in 2004, I should "
            "try to search for VIVA Media AG acquisitions or mergers to see if their new acronym was "
            "related to that. \nAction: Search[VIVA Media AG acquisitions]\nObservation: Could not find "
            "[VIVA Media AG acquisitions]. Similar: ['List of mergers and acquisitions by Alphabet', "
            "'List of mergers and acquisitions by Apple', 'List of mergers and acquisitions by Microsoft', "
            "'List of mergers and acquisitions by Facebook', 'List of mergers and acquisitions by Amazon', "
            "'List of mergers and acquisitions by IBM', 'List of mergers and acquisitions by Cisco Systems',"
            " 'List of mergers and acquisitions by Oracle', 'List of mergers and acquisitions by SAP', "
            "'List of mergers and acquisitions by Yahoo!']\nThought: Since I couldn't find information on "
            "acquisitions, I should try searching for VIVA Media AG partnerships or collaborations to see "
            "if their new acronym was related to that.\nAction: Search[VIVA Media AG partnerships]\n"
            "Observation: Could not find [VIVA Media AG partnerships]. Similar: ['List of airline codes', "
            "'List of country codes on British diplomatic vehicle registration plates', 'List of "
            "international vehicle registration codes', 'Vehicle registration plates of the United "
            "States for the diplomatic corps', 'Vehicle registration plates of the European Union', "
            "'List of diplomatic missions of Japan', 'List of diplomatic missions of Australia', "
            "'Diplomatic missions of the European Union', 'Vehicle registration plates of the United "
            "Kingdom', 'Vehicle registration plates of the United States']\nThought: I am still "
            "unable to find relevant information, I should try a broader search term like VIVA Media "
            "AG history to see if I can find any details about their name change and new acronym.\n"
            "Action: Search[VIVA Media AG history]"
        )
    ]
    reflections = (
        "You have attempted to answer the following question before and failed. Below is the last trial "
        "you attempted to answer the question.\nQuestion: VIVA Media AG changed it's name in 2004. What "
        "does their new acronym stand for?\nThought: I need to search for VIVA Media AG and find out their "
        "new acronym after changing their name in 2004. \nAction: Search[VIVA Media AG]\nObservation 1: "
        "Could not find [VIVA Media AG]. Similar: ['MTV Music (Polish TV channel)', 'Paramount International "
        "Networks', 'VIVA Plus', 'VIVA (German TV channel)', 'Viacom (1952–2005)', 'Vauxhall Viva', "
        "'GfK Entertainment charts', 'Lindt', 'Spellbound Entertainment', 'List of multinational "
        "corporations']\nThought: I should try searching for VIVA Media AG in a different way. Let me "
        "search for VIVA Media AG name change 2004. \nAction: Search[VIVA Media AG name change 2004]\n"
        "Observation 2: Could not find [VIVA Media AG name change 2004]. Similar: ['Vauxhall Viva', "
        "'GfK Entertainment charts', 'Opel Astra', 'About You Now', 'Puma (brand)', 'Priscilla Presley', "
        "'Bosch (company)', 'Schneider Electric', 'Sildenafil', 'Daihatsu Mira']\nThought: Given that I "
        "am unable to find information on VIVA Media AG and their name change in 2004, I should try to "
        "search for VIVA Media AG rebranding or any other related keywords to see if I can find the "
        "information. \nAction: Search[VIVA Media AG rebranding]\nObservation 3: Could not find "
        "[VIVA Media AG rebranding]. Similar: ['Paramount International Networks', 'Virgin Interactive "
        "Entertainment', 'Lake Las Vegas', 'Viacom (1952–2005)', 'Virgin Money UK plc', 'Voice of America',"
        " '2016 in Philippine television', 'PolyGram', 'Universal Music Group', 'Veolia Transport']\n"
        "(END PREVIOUS TRIAL)\n"
    )
    scratchpad = (
        "\nThought: I need to search for VIVA Media AG and find out their new acronym after changing "
        "their name in 2004. \nAction: Search[VIVA Media AG]\nObservation 1: Could not find [VIVA Media AG]. "
        "Similar: ['MTV Music (Polish TV channel)', 'Paramount International Networks', 'VIVA Plus', "
        "'VIVA (German TV channel)', 'Viacom (1952–2005)', 'Vauxhall Viva', 'GfK Entertainment charts', "
        "'Lindt', 'Spellbound Entertainment', 'List of multinational corporations']\nThought: I should try "
        "searching for VIVA Media AG in a different way. Let me search for VIVA Media AG name change 2004. "
        "\nAction: Search[VIVA Media AG name change 2004]\nObservation 2: Could not find [VIVA Media AG name "
        "change 2004]. Similar: ['Vauxhall Viva', 'GfK Entertainment charts', 'Opel Astra', 'About You Now', "
        "'Puma (brand)', 'Priscilla Presley', 'Bosch (company)', 'Schneider Electric', 'Sildenafil', "
        "'Daihatsu Mira']\nThought: Given that I am unable to find information on VIVA Media AG and "
        "their name change in 2004, I should try to search for VIVA Media AG rebranding or any other "
        "related keywords to see if I can find the information. \nAction: Search[VIVA Media AG rebranding]"
        "\nObservation 3: Could not find [VIVA Media AG rebranding]. Similar: ['Paramount International "
        "Networks', 'Virgin Interactive Entertainment', 'Lake Las Vegas', 'Viacom (1952–2005)', "
        "'Virgin Money UK plc', 'Voice of America', '2016 in Philippine television', 'PolyGram', "
        "'Universal Music Group', 'Veolia Transport']\nThought:"
    )
    gt_out = (
        "Given the lack of information on VIVA Media AG and their name change in 2004, "
        "I should try to search for VIVA Media AG acquisitions or mergers to see if their new acronym "
        "was related to that. Action: Search[VIVA Media AG acquisitions]Observation: Could not find "
        "[VIVA Media AG acquisitions]. Similar: ['List of mergers and acquisitions by Alphabet', "
        "'List of mergers and acquisitions by Apple', 'List of mergers and acquisitions by Microsoft', "
        "'List of mergers and acquisitions by Facebook', 'List of mergers and acquisitions by Amazon', "
        "'List of mergers and acquisitions by IBM', 'List of mergers and acquisitions by Cisco Systems', "
        "'List of mergers and acquisitions by Oracle', 'List of mergers and acquisitions by SAP', "
        "'List of mergers and acquisitions by Yahoo!']Thought: Since I couldn't find information on "
        "acquisitions, I should try searching for VIVA Media AG partnerships or collaborations to see "
        "if their new acronym was related to that.Action: Search[VIVA Media AG partnerships]"
        "Observation: Could not find [VIVA Media AG partnerships]. Similar: ['List of airline codes', "
        "'List of country codes on British diplomatic vehicle registration plates', "
        "'List of international vehicle registration codes', 'Vehicle registration plates of the "
        "United States for the diplomatic corps', 'Vehicle registration plates of the European Union', "
        "'List of diplomatic missions of Japan', 'List of diplomatic missions of Australia', "
        "'Diplomatic missions of the European Union', 'Vehicle registration plates of the United Kingdom', "
        "'Vehicle registration plates of the United States']Thought: I am still unable to find relevant "
        "information, I should try a broader search term like VIVA Media AG history to see if I can "
        "find any details about their name change and new acronym.Action: Search[VIVA Media AG history]"
    )
    out = _prompt_react_agent(
        llm=FakeListChatModel(responses=responses),
        examples=REACT_WEBTHINK_SIMPLE6_FEWSHOT_EXAMPLES,
        reflections=reflections,
        question=q,
        scratchpad=scratchpad,
        max_steps=6,
    )
    assert out == gt_out


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
        "",
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
        "",
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
        "",
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
        "",
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
        "",
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
        "",
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
        "",
        10,
        1603,
        gpt3_5_turbo_enc,
        "{question} {scratchpad} {examples} {max_steps}",
    )


def test__build_react_reflection_prompt() -> None:
    """Test _build_react_reflection_prompt function."""
    gt_out = 'You are an advanced reasoning agent that can improve based on self refection. You will be given a previous reasoning trial in which you were given access to an Docstore API environment and a question to answer. You were unsuccessful in answering the question either because you guessed the wrong answer with Finish[<answer>], or you used up your set number of reasoning steps. In a few sentences, Diagnose a possible reason for failure and devise a new, concise, high level plan that aims to mitigate the same failure. Use complete sentences.  \nHere are some examples:\n\n(END OF EXAMPLES)\n\nPrevious trial:\nQuestion: \n\nReflection:'
    out = _build_react_reflection_prompt(examples="", question="", scratchpad="")
    assert out == gt_out


def test__prompt_react_reflection() -> None:
    """Test _prompt_react_reflection function."""
    q = "VIVA Media AG changed it's name in 2004. What does their new acronym stand for?"

    # Test empty.
    out = _prompt_react_reflection(
        llm=FakeListChatModel(responses=["1"]),
        examples="",
        question="",
        scratchpad="",
    )
    assert isinstance(out, str)
    assert out == "1"

    # Test simple case.
    scratchpad = (
        "\nThought: I need to search for VIVA Media AG to find out their new acronym and what it stands for.\nAction: Search[VIVA Media AG]\n"
        "Observation 1: Could not find [VIVA Media AG]. Similar: ['MTV Music (Polish TV channel)', 'VIVA Plus', 'Paramount International Networks', "
        "'VIVA (German TV channel)', 'Viacom (1952–2005)', 'Vauxhall Viva', 'GfK Entertainment charts', 'Lindt', 'Spellbound Entertainment', "
        "'Ag-gag']\nThought: The search did not return exact information on VIVA Media AG. I should try searching for VIVA Media AG name change "
        "in 2004 to get more specific results.\nAction: Search[VIVA Media AG name change 2004]\nObservation 2: Could not find [VIVA Media AG "
        "name change 2004]. Similar: ['Vauxhall Viva', 'GfK Entertainment charts', 'Opel Astra', 'Puma (brand)', 'About You Now', 'Priscilla Presley', "
        "'Altium', 'Sildenafil', 'Bosch (company)', 'Schneider Electric']\nThought: I should try searching for the history of VIVA Media AG to see "
        "if I can find information on their name change in 2004 and their new acronym.\nAction: Search[history of VIVA Media AG]\n"
        "Observation 3: Could not find [history of VIVA Media AG]. Similar: ['MTV Music (Polish TV channel)', 'VIVA Plus', 'VIVA (German TV channel)', "
        "'Vauxhall Viva', 'Lindt', 'GfK Entertainment charts', 'Spellbound Entertainment', 'Ag-gag', 'Springer Publishing', 'Kimberly-Clark']"
    )
    responses = [
        (
            "The failure in this reasoning trial was due to not being able to find specific information on VIVA Media AG and its name change "
            "in 2004. To mitigate this failure, the agent should consider broadening the search terms to include related keywords such as "
            '"corporate rebranding" or "corporate name change" in addition to the specific company name. This will help in obtaining more '
            "relevant and specific results that may provide the necessary information to answer the question accurately."
        )
    ]
    gt_out = (
        "The failure in this reasoning trial was due to not being able to find specific information on VIVA Media AG and its name change in 2004. "
        'To mitigate this failure, the agent should consider broadening the search terms to include related keywords such as "corporate rebranding" '
        'or "corporate name change" in addition to the specific company name. This will help in obtaining more relevant and specific results that may '
        "provide the necessary information to answer the question accurately."
    )
    out = _prompt_react_reflection(
        llm=FakeListChatModel(responses=responses),
        examples=REFLEXION_REACT_REFLECT_FEWSHOT_EXAMPLES,
        question=q,
        scratchpad=scratchpad,
    )
    assert out == gt_out


def test_react_reflect_last_attempt() -> None:
    """Test react_reflect_last_attempt function."""
    scratchpad = ""
    out = react_reflect_last_attempt(scratchpad)
    assert out == [""]


def test_react_reflect_reflexion() -> None:
    """Test react_reflect_reflexion function."""
    out = react_reflect_reflexion(
        llm=FakeListChatModel(responses=["1"]),
        reflections=[""],
        examples="",
        question="",
        scratchpad="",
    )
    assert isinstance(out, list)
    assert out == ["", "1"]


def test_react_reflect_last_attempt_and_reflexion() -> None:
    """Test react_reflect_last_attempt_and_reflexion function."""
    out = react_reflect_last_attempt_and_reflexion(
        llm=FakeListChatModel(responses=["1"]),
        examples="",
        question="",
        scratchpad="",
    )
    assert isinstance(out, list)
    assert out == ["1"]


def test_react_reflect() -> None:
    """Test react_reflect function."""
    # Invalid strategy.
    with pytest.raises(NotImplementedError):
        out = react_reflect(
            strategy="invalid input",
            llm=FakeListChatModel(responses=["1"]),
            reflections=[""],
            examples="",
            question="",
            scratchpad="",
        )

    # Last attempt.
    out = react_reflect(
        strategy="last_attempt",
        llm=FakeListChatModel(responses=["1"]),
        reflections=[""],
        examples="",
        question="",
        scratchpad="",
    )
    assert out == [""]

    # Reflexion.
    out = react_reflect(
        strategy="reflexion",
        llm=FakeListChatModel(responses=["1"]),
        reflections=[""],
        examples="",
        question="",
        scratchpad="",
    )
    assert isinstance(out, list)
    assert out == ["", "1"]

    # Last attempt and Reflexion.
    out = react_reflect(
        strategy="last_attempt_and_reflexion",
        llm=FakeListChatModel(responses=["1"]),
        reflections=[""],
        examples="",
        question="",
        scratchpad="",
    )
    assert isinstance(out, list)
    assert out == ["1"]
