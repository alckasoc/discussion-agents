"""Unit tests for Reflexion QA strategies."""

from langchain_community.chat_models.fake import FakeListChatModel
from langchain_core.language_models.chat_models import BaseChatModel

from agential.cog.modules.reflect.reflexion import (
    ReflexionCoTReflector,
    ReflexionReActReflector,
)
from agential.cog.prompts.agent.reflexion import (
    HOTPOTQA_FEWSHOT_EXAMPLES_REFLEXION_COT_REFLECT,
    HOTPOTQA_FEWSHOT_EXAMPLES_REFLEXION_REACT_REFLECT,
    REFLEXION_COT_INSTRUCTION_HOTPOTQA,
    REFLEXION_COT_REFLECT_INSTRUCTION_HOTPOTQA,
    REFLEXION_REACT_INSTRUCTION_HOTPOTQA,
    REFLEXION_REACT_REFLECT_INSTRUCTION_HOTPOTQA,
)
from agential.cog.prompts.benchmark.hotpotqa import (
    HOTPOTQA_FEWSHOT_EXAMPLES_COT,
    HOTPOTQA_FEWSHOT_EXAMPLES_REACT,
)
from agential.cog.strategies.reflexion.qa import (
    ReflexionCoTAmbigNQStrategy,
    ReflexionCoTFEVERStrategy,
    ReflexionCoTHotQAStrategy,
    ReflexionCoTQAStrategy,
    ReflexionCoTTriviaQAStrategy,
    ReflexionReActAmbigNQStrategy,
    ReflexionReActFEVERStrategy,
    ReflexionReActHotQAStrategy,
    ReflexionReActQAStrategy,
    ReflexionReActTriviaQAStrategy,
    parse_qa_action,
)


def test_parse_qa_action() -> None:
    """Tests parse_qa_action."""
    action = "Calculate[sum = 4 + 6]"
    action_type, argument = parse_qa_action(action)
    assert action_type == "Calculate"
    assert argument == "sum = 4 + 6"

    action = "Finish[result = 7 - 2]"
    action_type, argument = parse_qa_action(action)
    assert action_type == "Finish"
    assert argument == "result = 7 - 2"

    action = "InvalidAction[result = 10 / 2]"
    action_type, argument = parse_qa_action(action)
    assert action_type == "InvalidAction"
    assert argument == "result = 10 / 2"

    action = "NoBrackets"
    action_type, argument = parse_qa_action(action)
    assert action_type == ""
    assert argument == ""

    action = "EmptyBrackets[]"
    action_type, argument = parse_qa_action(action)
    assert action_type == ""
    assert argument == ""


def test_reflexion_cot_init() -> None:
    """Test ReflexionCoTQAStrategy initialization."""
    llm = FakeListChatModel(responses=[])
    strategy = ReflexionCoTQAStrategy(llm=llm)
    assert isinstance(strategy.llm, BaseChatModel)
    assert isinstance(strategy.reflector, ReflexionCoTReflector)
    assert strategy.max_reflections == 3
    assert strategy.max_trials == 1
    assert strategy._scratchpad == ""
    assert strategy._finished == False
    assert strategy._answer == ""


def test_reflexion_cot_generate() -> None:
    """Tests ReflexionCoTQAStrategy generate."""
    question = "VIVA Media AG changed it's name in 2004. What does their new acronym stand for?"

    gt_scratchpad = '\nThought: The question is asking for the acronym that VIVA Media AG changed its name to in 2004. Based on the context, I know that VIVA Media AG is now known as VIVA Media GmbH. Therefore, the acronym "GmbH" stands for "Gesellschaft mit beschränkter Haftung" in German, which translates to "company with limited liability" in English.'
    gt_out = 'The question is asking for the acronym that VIVA Media AG changed its name to in 2004. Based on the context, I know that VIVA Media AG is now known as VIVA Media GmbH. Therefore, the acronym "GmbH" stands for "Gesellschaft mit beschränkter Haftung" in German, which translates to "company with limited liability" in English.'
    responses = [
        'The question is asking for the acronym that VIVA Media AG changed its name to in 2004. Based on the context, I know that VIVA Media AG is now known as VIVA Media GmbH. Therefore, the acronym "GmbH" stands for "Gesellschaft mit beschränkter Haftung" in German, which translates to "company with limited liability" in English.',
    ]
    llm = FakeListChatModel(responses=responses)
    strategy = ReflexionCoTQAStrategy(llm=llm)
    out = strategy.generate(
        question=question,
        examples=HOTPOTQA_FEWSHOT_EXAMPLES_COT,
        reflections="",
        prompt=REFLEXION_COT_INSTRUCTION_HOTPOTQA,
        additional_keys={},
    )
    assert out == gt_out
    assert strategy._scratchpad == gt_scratchpad
    assert strategy._finished == False
    assert strategy._answer == ""


def test_reflexion_cot_generate_action() -> None:
    """Tests ReflexionCoTQAStrategy generate_action."""
    question = "VIVA Media AG changed it's name in 2004. What does their new acronym stand for?"

    responses = ["Finish[Verwaltung von Internet Video und Audio]"]
    llm = FakeListChatModel(responses=responses)
    strategy = ReflexionCoTQAStrategy(llm=llm)
    action_type, query = strategy.generate_action(
        question=question,
        examples=HOTPOTQA_FEWSHOT_EXAMPLES_COT,
        reflections="",
        prompt=REFLEXION_COT_INSTRUCTION_HOTPOTQA,
        additional_keys={},
    )
    assert action_type == "Finish"
    assert query == "Verwaltung von Internet Video und Audio"
    assert strategy._finished == False
    assert strategy._answer == ""
    assert (
        strategy._scratchpad
        == "\nAction: Finish[Verwaltung von Internet Video und Audio]"
    )


def test_reflexion_cot_generate_observation() -> None:
    """Tests ReflexionCoTQAStrategy generate_observation."""
    # Case 1: action_type is "Finish" and answer is correct.
    llm = FakeListChatModel(responses=[])
    strategy = ReflexionCoTQAStrategy(llm=llm)
    is_correct, obs = strategy.generate_observation(
        action_type="Finish",
        query="correct_answer",
        key="correct_answer",
    )
    assert is_correct == True
    assert obs == "Answer is CORRECT"
    assert "Observation: Answer is CORRECT" in strategy._scratchpad

    # Case 2: action_type is "Finish" and answer is incorrect.
    strategy = ReflexionCoTQAStrategy(llm=llm)
    is_correct, obs = strategy.generate_observation(
        action_type="Finish",
        query="incorrect_answer",
        key="correct_answer",
    )
    assert is_correct == False
    assert obs == "Answer is INCORRECT"
    assert "Observation: Answer is INCORRECT" in strategy._scratchpad

    # Case 3: action_type is not "Finish".
    strategy = ReflexionCoTQAStrategy(llm=llm)
    is_correct, obs = strategy.generate_observation(
        action_type="Calculate",
        query="some_query",
        key="correct_answer",
    )
    assert is_correct == False
    assert obs == "Invalid action type, please try again."
    assert "Observation: Invalid action type, please try again." in strategy._scratchpad


def test_reflexion_cot_create_output_dict() -> None:
    """Tests ReflexionCoTQAStrategy create_output_dict."""
    strategy = ReflexionCoTQAStrategy(llm=FakeListChatModel(responses=[]))

    # Setting a dummy answer for testing.
    strategy._answer = "correct_answer"

    # Test case 1: Correct answer.
    output = strategy.create_output_dict(
        thought="This is a thought.",
        action_type="Finish",
        obs="Observation: Answer is CORRECT",
        is_correct=True,
        reflections=[],
    )
    expected_output = {
        "thought": "This is a thought.",
        "action_type": "Finish",
        "observation": "Observation: Answer is CORRECT",
        "answer": "correct_answer",
        "is_correct": True,
        "reflections": [],
    }
    assert output == expected_output

    # Test case 2: Incorrect answer.
    strategy._answer = "incorrect_answer"
    output = strategy.create_output_dict(
        thought="This is a thought.",
        action_type="Finish",
        obs="Observation: Answer is INCORRECT",
        is_correct=False,
        reflections=[],
    )
    expected_output = {
        "thought": "This is a thought.",
        "action_type": "Finish",
        "observation": "Observation: Answer is INCORRECT",
        "answer": "incorrect_answer",
        "is_correct": False,
        "reflections": [],
    }
    assert output == expected_output

    # Test case 3: Invalid action type.
    strategy._answer = "some_answer"
    output = strategy.create_output_dict(
        thought="This is another thought.",
        action_type="Calculate",
        obs="Observation: Invalid action type, please try again.",
        is_correct=False,
        reflections=[],
    )
    expected_output = {
        "thought": "This is another thought.",
        "action_type": "Calculate",
        "observation": "Observation: Invalid action type, please try again.",
        "answer": "some_answer",
        "is_correct": False,
        "reflections": [],
    }
    assert output == expected_output


def test_reflexion_cot_halting_condition() -> None:
    """Tests ReflexionCoTQAStrategy halting_condition."""
    llm = FakeListChatModel(responses=[])
    strategy = ReflexionCoTQAStrategy(llm=llm, max_trials=3)

    strategy._answer = "incorrect_answer"
    assert strategy.halting_condition(3, "correct_answer") == True

    strategy._answer = "correct_answer"
    assert strategy.halting_condition(2, "correct_answer") == True

    strategy._answer = "incorrect_answer"
    assert strategy.halting_condition(2, "correct_answer") == False


def test_reflexion_cot_reset() -> None:
    """Tests ReflexionCoTQAStrategy reset."""
    llm = FakeListChatModel(responses=[])
    strategy = ReflexionCoTQAStrategy(llm=llm, max_trials=3)

    strategy._scratchpad = "Initial scratchpad content"
    strategy._finished = True
    strategy._answer = "Some answer"

    # Test case 1: Reset everything.
    strategy.reset()
    assert strategy._scratchpad == ""
    assert strategy._finished == False
    assert strategy._answer == ""

    strategy._scratchpad = "Initial scratchpad content"
    strategy._finished = True
    strategy._answer = "Some answer"

    # Test case 2: Reset only scratchpad.
    strategy.reset(only_scratchpad=True)
    assert strategy._scratchpad == ""
    assert strategy._finished == True
    assert strategy._answer == "Some answer"


def test_reflexion_cot_reflect() -> None:
    """Tests ReflexionCoTQAStrategy reflect."""
    question = "VIVA Media AG changed it's name in 2004. What does their new acronym stand for?"

    llm = FakeListChatModel(responses=[])
    strategy = ReflexionCoTQAStrategy(llm=llm, max_trials=3)

    gt_out = "You have attempted to answer the following question before and failed. Below is the last trial you attempted to answer the question.\nQuestion: VIVA Media AG changed it's name in 2004. What does their new acronym stand for?\n\n(END PREVIOUS TRIAL)\n"
    _, out = strategy.reflect(
        reflect_strategy="last_attempt",
        question=question,
        examples=HOTPOTQA_FEWSHOT_EXAMPLES_REFLEXION_COT_REFLECT,
        prompt=REFLEXION_COT_REFLECT_INSTRUCTION_HOTPOTQA,
        additional_keys={},
    )
    assert out == gt_out


def test_reflexion_cot_reflect_condition() -> None:
    """Tests ReflexionCoTQAStrategy reflect_condition."""
    llm = FakeListChatModel(responses=[])
    strategy = ReflexionCoTQAStrategy(llm)

    assert not strategy.reflect_condition(0, "strategy1", "key1")
    assert strategy.reflect_condition(1, "strategy1", "key1")
    assert strategy.reflect_condition(1, "strategy1", "key2")
    assert strategy.reflect_condition(1, "", "key2")


def test_reflexion_cot_instantiate_strategies() -> None:
    """Test instantiate all ReflexionCoT QA strategies."""
    llm = FakeListChatModel(responses=[])
    hotqa_strategy = ReflexionCoTHotQAStrategy(llm=llm)
    triviaqa_strategy = ReflexionCoTTriviaQAStrategy(llm=llm)
    ambignq_strategy = ReflexionCoTAmbigNQStrategy(llm=llm)
    fever_strategy = ReflexionCoTFEVERStrategy(llm=llm)

    assert isinstance(hotqa_strategy, ReflexionCoTHotQAStrategy)
    assert isinstance(triviaqa_strategy, ReflexionCoTTriviaQAStrategy)
    assert isinstance(ambignq_strategy, ReflexionCoTAmbigNQStrategy)
    assert isinstance(fever_strategy, ReflexionCoTFEVERStrategy)


def test_reflexion_react_init() -> None:
    """Test ReflexionReActStrategy initialization."""
    llm = FakeListChatModel(responses=[])
    strategy = ReflexionReActQAStrategy(llm=llm)
    assert isinstance(strategy.llm, BaseChatModel)
    assert isinstance(strategy.reflector, ReflexionReActReflector)
    assert strategy.max_reflections == 3
    assert strategy.max_trials == 1
    assert strategy._scratchpad == ""
    assert strategy._finished == False
    assert strategy._answer == ""


def test_reflexion_react_generate() -> None:
    """Tests ReflexionReActQAStrategy generate."""
    question = "VIVA Media AG changed it's name in 2004. What does their new acronym stand for?"

    gt_scratchpad = "\nThought: I need to search for VIVA Media AG and find out their new acronym after changing their name in 2004."
    gt_out = "I need to search for VIVA Media AG and find out their new acronym after changing their name in 2004."
    responses = [
        "I need to search for VIVA Media AG and find out their new acronym after changing their name in 2004."
    ]
    llm = FakeListChatModel(responses=responses)
    strategy = ReflexionReActQAStrategy(llm=llm)
    out = strategy.generate(
        question=question,
        examples=HOTPOTQA_FEWSHOT_EXAMPLES_REACT,
        reflections="",
        prompt=REFLEXION_REACT_INSTRUCTION_HOTPOTQA,
        additional_keys={},
        max_steps=5,
    )
    assert out == gt_out
    assert strategy._scratchpad == gt_scratchpad


def test_reflexion_react_generate_action() -> None:
    """Tests ReflexionReActQAStrategy generate_action."""
    question = "VIVA Media AG changed it's name in 2004. What does their new acronym stand for?"

    gt_scratchpad = "\nAction: Search[VIVA Media AG]"
    responses = [
        "Search[VIVA Media AG]",
    ]
    llm = FakeListChatModel(responses=responses)
    strategy = ReflexionReActQAStrategy(llm=llm)
    action_type, query = strategy.generate_action(
        question=question,
        examples=HOTPOTQA_FEWSHOT_EXAMPLES_REACT,
        reflections="",
        prompt=REFLEXION_REACT_INSTRUCTION_HOTPOTQA,
        additional_keys={},
        max_steps=5,
    )
    assert action_type == "Search"
    assert query == "VIVA Media AG"
    assert strategy._scratchpad == gt_scratchpad


def test_reflexion_react_generate_observation() -> None:
    """Tests ReflexionReActQAStrategy generate_observation."""
    llm = FakeListChatModel(responses=[])
    strategy = ReflexionReActQAStrategy(llm=llm)
    strategy.docstore.search = lambda x: "Search result"
    is_correct, obs, external_tool_info = strategy.generate_observation(
        step_idx=1,
        action_type="Search",
        query="VIVA Media AG",
        key="key1",
    )
    assert not is_correct
    assert isinstance(obs, str)
    assert strategy._scratchpad != ""
    assert not strategy._finished
    assert strategy._answer == ""
    assert external_tool_info == {"search_result": "Search result", "lookup_result": ""}

    strategy.docstore.lookup = lambda x: "Lookup result"
    is_correct, obs, external_tool_info = strategy.generate_observation(
        step_idx=1,
        action_type="Lookup",
        query="VIVA Media AG",
        key="key1",
    )
    assert not is_correct
    assert isinstance(obs, str)
    assert strategy._scratchpad != ""
    assert not strategy._finished
    assert strategy._answer == ""
    assert external_tool_info == {"search_result": "", "lookup_result": "Lookup result"}

    is_correct, obs, external_tool_info = strategy.generate_observation(
        step_idx=1,
        action_type="Finish",
        query="VIVA Media AG",
        key="key1",
    )
    assert not is_correct
    assert isinstance(obs, str)
    assert strategy._scratchpad != ""
    assert strategy._finished
    assert strategy._answer == "VIVA Media AG"
    assert external_tool_info == {"search_result": "", "lookup_result": ""}


def test_reflexion_react_create_output_dict() -> None:
    """Tests ReflexionReActQAStrategy create_output_dict."""
    strategy = ReflexionReActQAStrategy(llm=FakeListChatModel(responses=[]))

    # Test case 1: Valid output creation
    react_out = [
        {
            "thought": "First thought",
            "action_type": "Query",
            "query": "What is the capital of France?",
            "observation": "Observation: Answer is CORRECT",
            "is_correct": True,
        }
    ]
    reflections = "Reflection on the first thought."
    output = strategy.create_output_dict(react_out, reflections)
    expected_output = {
        "react_output": react_out,
        "reflections": reflections,
    }
    assert output == expected_output

    # Test case 2: Multiple steps in react_out
    react_out = [
        {
            "thought": "First thought",
            "action_type": "Query",
            "query": "What is the capital of France?",
            "observation": "Observation: Answer is CORRECT",
            "is_correct": True,
        },
        {
            "thought": "Second thought",
            "action_type": "Validate",
            "query": "Is 2+2=4?",
            "observation": "Observation: Answer is CORRECT",
            "is_correct": True,
        },
    ]
    reflections = "Reflection on the second thought."
    output = strategy.create_output_dict(react_out, reflections)
    expected_output = {
        "react_output": react_out,
        "reflections": reflections,
    }
    assert output == expected_output

    # Test case 3: Empty react_out
    react_out = []
    reflections = "No reflections since no actions were taken."
    output = strategy.create_output_dict(react_out, reflections)
    expected_output = {
        "react_output": react_out,
        "reflections": reflections,
    }
    assert output == expected_output


def test_reflexion_react_react_create_output_dict() -> None:
    """Tests ReflexionReActQAStrategy react_create_output_dict."""
    strategy = ReflexionReActQAStrategy(llm=FakeListChatModel(responses=[]))

    # Test case 1: Valid output creation
    output = strategy.react_create_output_dict(
        thought="Initial thought",
        action_type="Query",
        query="What is the capital of France?",
        obs="Observation: Answer is CORRECT",
        external_tool_info={"search_result": "", "lookup_result": ""},
        is_correct=True,
    )
    expected_output = {
        "thought": "Initial thought",
        "action_type": "Query",
        "query": "What is the capital of France?",
        "observation": "Observation: Answer is CORRECT",
        "answer": "",
        "external_tool_info": {"search_result": "", "lookup_result": ""},
        "is_correct": True,
    }
    assert output == expected_output

    # Test case 2: Another valid output creation
    output = strategy.react_create_output_dict(
        thought="Second thought",
        action_type="Validate",
        query="Is 2+2=4?",
        obs="Observation: Answer is CORRECT",
        external_tool_info={"search_result": "", "lookup_result": ""},
        is_correct=True,
    )
    expected_output = {
        "thought": "Second thought",
        "action_type": "Validate",
        "query": "Is 2+2=4?",
        "observation": "Observation: Answer is CORRECT",
        "answer": "",
        "external_tool_info": {"search_result": "", "lookup_result": ""},
        "is_correct": True,
    }
    assert output == expected_output

    # Test case 3: Incorrect answer handling
    output = strategy.react_create_output_dict(
        thought="Final thought",
        action_type="Answer",
        query="What is the square root of 16?",
        obs="Observation: Answer is INCORRECT",
        external_tool_info={"search_result": "", "lookup_result": ""},
        is_correct=False,
    )
    expected_output = {
        "thought": "Final thought",
        "action_type": "Answer",
        "query": "What is the square root of 16?",
        "observation": "Observation: Answer is INCORRECT",
        "answer": "",
        "external_tool_info": {"search_result": "", "lookup_result": ""},
        "is_correct": False,
    }
    assert output == expected_output


def test_reflexion_react_halting_condition() -> None:
    """Tests ReflexionReActQAStrategy halting_condition."""
    llm = FakeListChatModel(responses=[])

    # Test case 1: Halting condition met because answer is incorrect and index is less than max_trials.
    strategy = ReflexionReActQAStrategy(llm=llm, max_trials=5)
    strategy._answer = "incorrect_answer"
    assert strategy.halting_condition(3, "correct_answer") == False

    # Test case 2: Halting condition not met because answer is correct.
    strategy = ReflexionReActQAStrategy(llm=llm, max_trials=5)
    strategy._answer = "correct_answer"
    assert strategy.halting_condition(3, "correct_answer") == True

    # Test case 3: Halting condition not met because index is greater than or equal to max_trials.
    strategy = ReflexionReActQAStrategy(llm=llm, max_trials=3)
    strategy._answer = "incorrect_answer"
    assert strategy.halting_condition(4, "correct_answer") == True

    # Test case 4: Halting condition met using max_trials from kwargs.
    strategy = ReflexionReActQAStrategy(llm=llm, max_trials=5)
    strategy._answer = "incorrect_answer"
    assert strategy.halting_condition(3, "correct_answer", max_trials=4) == False

    # Test case 5: Halting condition not met using max_trials from kwargs.
    strategy = ReflexionReActQAStrategy(llm=llm, max_trials=5)
    strategy._answer = "incorrect_answer"
    assert strategy.halting_condition(4, "correct_answer", max_trials=3) == True


def test_reflexion_react_react_halting_condition() -> None:
    """Tests ReflexionReActQAStrategy react_halting_condition."""
    strategy = ReflexionReActQAStrategy(llm=FakeListChatModel(responses=[]))

    idx = 0
    question = "What is the capital of France?"
    examples = ""
    reflections = ""
    prompt = "Answer the question."

    assert not strategy.react_halting_condition(
        idx, question, examples, reflections, prompt, {}
    )


def test_reflexion_react_reset() -> None:
    """Tests ReflexionReActQAStrategy reset."""
    llm = FakeListChatModel(responses=[])
    strategy = ReflexionReActQAStrategy(llm=llm)
    strategy._scratchpad = "Some previous state"
    strategy._finished = True

    strategy.reset()

    assert strategy._scratchpad == ""
    assert not strategy._finished


def test_reflexion_react_reflect() -> None:
    """Tests ReflexionReActQAStrategy reflect."""
    question = "VIVA Media AG changed it's name in 2004. What does their new acronym stand for?"

    gt_reflections = "You have attempted to answer following question before and failed. The following reflection(s) give a plan to avoid failing to answer the question in the same way you did previously. Use them to improve your strategy of correctly answering the given question.\nReflections:\n- 1"
    llm = FakeListChatModel(responses=["1"])
    strategy = ReflexionReActQAStrategy(llm=llm)
    _, reflections = strategy.reflect(
        reflect_strategy="reflexion",
        question=question,
        examples=HOTPOTQA_FEWSHOT_EXAMPLES_REFLEXION_REACT_REFLECT,
        prompt=REFLEXION_REACT_REFLECT_INSTRUCTION_HOTPOTQA,
        additional_keys={},
    )
    assert reflections == gt_reflections


def test_reflexion_react_reflect_condition() -> None:
    """Tests ReflexionReActQAStrategy reflect_condition."""
    llm = FakeListChatModel(responses=["1"])
    strategy = ReflexionReActQAStrategy(llm=llm)
    out = strategy.reflect_condition(
        step_idx=1,
        reflect_strategy="reflexion",
        question="VIVA Media AG changed it's name in 2004. What does their new acronym stand for?",
        examples=HOTPOTQA_FEWSHOT_EXAMPLES_REFLEXION_REACT_REFLECT,
        key="key",
        prompt=REFLEXION_REACT_REFLECT_INSTRUCTION_HOTPOTQA,
        additional_keys={},
    )
    assert not out


def test_reflexion_react_instantiate_strategies() -> None:
    """Test instantiate all ReflexionReAct QA strategies."""
    llm = FakeListChatModel(responses=[])
    hotqa_strategy = ReflexionReActHotQAStrategy(llm=llm)
    triviaqa_strategy = ReflexionReActTriviaQAStrategy(llm=llm)
    ambignq_strategy = ReflexionReActAmbigNQStrategy(llm=llm)
    fever_strategy = ReflexionReActFEVERStrategy(llm=llm)

    assert isinstance(hotqa_strategy, ReflexionReActHotQAStrategy)
    assert isinstance(triviaqa_strategy, ReflexionReActTriviaQAStrategy)
    assert isinstance(ambignq_strategy, ReflexionReActAmbigNQStrategy)
    assert isinstance(fever_strategy, ReflexionReActFEVERStrategy)
