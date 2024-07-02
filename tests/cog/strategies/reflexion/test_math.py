"""Unit tests for Reflexion Math strategies."""

from langchain_community.chat_models.fake import FakeListChatModel
from langchain_core.language_models.chat_models import BaseChatModel

from agential.cog.modules.reflect.reflexion import (
    ReflexionCoTReflector,
    ReflexionReActReflector,
)
from agential.cog.prompts.agent.reflexion import (
    GSM8K_FEWSHOT_EXAMPLES_REFLEXION_COT_REFLECT,
    GSM8K_FEWSHOT_EXAMPLES_REFLEXION_REACT_REFLECT,
    REFLEXION_COT_INSTRUCTION_GSM8K,
    REFLEXION_COT_REFLECT_INSTRUCTION_GSM8K,
    REFLEXION_REACT_INSTRUCTION_GSM8K,
    REFLEXION_REACT_REFLECT_INSTRUCTION_GSM8K,
)
from agential.cog.prompts.benchmark.gsm8k import (
    GSM8K_FEWSHOT_EXAMPLES_COT,
    GSM8K_FEWSHOT_EXAMPLES_REACT,
)
from agential.cog.strategies.reflexion.math import (
    ReflexionCoTGSM8KStrategy,
    ReflexionCoTMathStrategy,
    ReflexionCoTSVAMPStrategy,
    ReflexionCoTTabMWPStrategy,
    ReflexionReActMathStrategy,
    ReflexionReActSVAMPStrategy,
    ReflexionReActTabMWPStrategy,
    parse_math_action_cot,
    parse_math_action_react,
)


def test_parse_math_action_cot() -> None:
    """Tests parse_math_action_cot."""
    action = "Finish the calculation```python\nresult = 5 + 3\n```"
    action_type, query = parse_math_action_cot(action)
    assert action_type == "Finish"
    assert query == "result = 5 + 3"

    action = "complete the task```python\nanswer = 10 * 2\n```"
    action_type, query = parse_math_action_cot(action)
    assert action_type == ""
    assert query == ""


def test_parse_math_action_react() -> None:
    """Tests parse_math_action_react."""
    action = "Calculate the sum```python\nsum = 4 + 6\n```"
    action_type, query = parse_math_action_react(action)
    assert action_type == "Calculate"
    assert query == "sum = 4 + 6"

    action = "Finish the operation```python\nresult = 7 - 2\n```"
    action_type, query = parse_math_action_react(action)
    assert action_type == "Finish"
    assert query == "result = 7 - 2"

    action = "complete the task```python\noutput = 10 / 2\n```"
    action_type, query = parse_math_action_react(action)
    assert action_type == ""
    assert query == ""


def test_reflexion_cot_init() -> None:
    """Tests ReflexionCoTMathStrategy init."""
    llm = FakeListChatModel(responses=[])
    strategy = ReflexionCoTGSM8KStrategy(llm=llm)
    assert isinstance(strategy.llm, BaseChatModel)
    assert isinstance(strategy.reflector, ReflexionCoTReflector)
    assert strategy.max_reflections == 3
    assert strategy.max_trials == 1
    assert strategy._scratchpad == ""
    assert strategy._finished == False
    assert strategy._answer == ""


def test_reflexion_cot_generate() -> None:
    """Tests ReflexionCoTMathStrategy generate."""
    question = "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with 4933828. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"

    gt_out = "Let's calculate the total number of eggs she sells after breakfast and baking muffins. Then, we can find out how much she makes daily at the farmers' market."
    gt_scratchpad = "\nThought: Let's calculate the total number of eggs she sells after breakfast and baking muffins. Then, we can find out how much she makes daily at the farmers' market."
    responses = [
        "Let's calculate the total number of eggs she sells after breakfast and baking muffins. Then, we can find out how much she makes daily at the farmers' market.\nAction: Finish[\n```python\neggs_per_day = 16\neggs_for_breakfast = 3\neggs_for_muffins = 4933828\ntotal_eggs_sold = eggs_per_day - eggs_for_breakfast - eggs_for_muffins\nprice_per_egg = 2\ndaily_income = total_eggs_sold * price_per_egg\nanswer = daily_income\n```\n]"
    ]
    llm = FakeListChatModel(responses=responses)
    strategy = ReflexionCoTMathStrategy(llm=llm)
    out = strategy.generate(
        question=question,
        examples=GSM8K_FEWSHOT_EXAMPLES_COT,
        reflections="",
        prompt=REFLEXION_COT_INSTRUCTION_GSM8K,
        additional_keys={},
    )
    assert out == gt_out
    assert strategy._scratchpad == gt_scratchpad
    assert strategy._finished == False
    assert strategy._answer == ""


def test_reflexion_cot_generate_action() -> None:
    """Tests ReflexionCoTMathStrategy generate_action."""
    question = "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with 4933828. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"

    responses = [
        "Finish[\n```python\neggs_laid_per_day = 16\neggs_eaten_for_breakfast = 3\neggs_used_for_muffins = 4933828\neggs_sold = eggs_laid_per_day - eggs_eaten_for_breakfast - eggs_used_for_muffins\nprice_per_egg = 2\nmoney_made_per_day = eggs_sold * price_per_egg\nanswer = money_made_per_day\n```\n]"
    ]
    llm = FakeListChatModel(responses=responses)
    strategy = ReflexionCoTMathStrategy(llm=llm)
    action_type, query = strategy.generate_action(
        question=question,
        examples=GSM8K_FEWSHOT_EXAMPLES_COT,
        reflections="",
        prompt=REFLEXION_COT_INSTRUCTION_GSM8K,
        additional_keys={},
    )
    assert action_type == "Finish"
    assert (
        query
        == "eggs_laid_per_day = 16\neggs_eaten_for_breakfast = 3\neggs_used_for_muffins = 4933828\neggs_sold = eggs_laid_per_day - eggs_eaten_for_breakfast - eggs_used_for_muffins\nprice_per_egg = 2\nmoney_made_per_day = eggs_sold * price_per_egg\nanswer = money_made_per_day"
    )
    assert strategy._finished == False
    assert strategy._answer == ""
    assert (
        strategy._scratchpad
        == "\nAction: Finish[\n```python\neggs_laid_per_day = 16\neggs_eaten_for_breakfast = 3\neggs_used_for_muffins = 4933828\neggs_sold = eggs_laid_per_day - eggs_eaten_for_breakfast - eggs_used_for_muffins\nprice_per_egg = 2\nmoney_made_per_day = eggs_sold * price_per_egg\nanswer = money_made_per_day\n```\n]"
    )


def test_reflexion_cot_generate_observation() -> None:
    """Tests ReflexionCoTMathStrategy generate_observation."""
    # Case 1: action_type is "Finish" and answer is correct.
    llm = FakeListChatModel(responses=[])
    strategy = ReflexionCoTMathStrategy(llm=llm)
    is_correct, obs = strategy.generate_observation(
        action_type="Finish",
        query="correct_answer",
        key="correct_answer",
    )
    assert is_correct == False
    assert obs == "Answer is INCORRECT"
    assert "Observation: Answer is INCORRECT" in strategy._scratchpad

    # Case 2: action_type is "Finish" and answer is incorrect.
    strategy = ReflexionCoTMathStrategy(llm=llm)
    is_correct, obs = strategy.generate_observation(
        action_type="Finish",
        query="incorrect_answer",
        key="correct_answer",
    )
    assert is_correct == False
    assert obs == "Answer is INCORRECT"
    assert "Observation: Answer is INCORRECT" in strategy._scratchpad

    # Case 3: action_type is not "Finish".
    strategy = ReflexionCoTMathStrategy(llm=llm)
    is_correct, obs = strategy.generate_observation(
        action_type="Calculate",
        query="some_query",
        key="correct_answer",
    )
    assert is_correct == False
    assert obs == "Invalid action type, please try again."
    assert "Observation: Invalid action type, please try again." in strategy._scratchpad


def test_reflexion_cot_create_output_dict() -> None:
    """Tests ReflexionCoTMathStrategy create_output_dict."""
    strategy = ReflexionCoTMathStrategy(llm=FakeListChatModel(responses=[]))

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


def test_reflexion_cot_halting_condition() -> None:
    """Tests ReflexionCoTMathStrategy halting_condition."""
    llm = FakeListChatModel(responses=[])
    strategy = ReflexionCoTMathStrategy(llm=llm, max_trials=3)

    strategy._answer = "incorrect_answer"
    assert strategy.halting_condition(3, "correct_answer") == True

    strategy._answer = "correct_answer"
    assert strategy.halting_condition(2, "correct_answer") == False

    strategy._answer = "incorrect_answer"
    assert strategy.halting_condition(2, "correct_answer") == False


def test_reflexion_cot_reset() -> None:
    """Tests ReflexionCoTMathStrategy reset."""
    llm = FakeListChatModel(responses=[])
    strategy = ReflexionCoTMathStrategy(llm=llm, max_trials=3)

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
    """Tests ReflexionCoTMathStrategy reflect."""
    question = "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with 4933828. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"

    llm = FakeListChatModel(responses=[])
    strategy = ReflexionCoTMathStrategy(llm=llm, max_trials=3)

    gt_out = "You have attempted to answer the following question before and failed. Below is the last trial you attempted to answer the question.\nQuestion: Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with 4933828. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?\n\n(END PREVIOUS TRIAL)\n"
    _, out = strategy.reflect(
        reflect_strategy="last_attempt",
        question=question,
        examples=GSM8K_FEWSHOT_EXAMPLES_REFLEXION_COT_REFLECT,
        prompt=REFLEXION_COT_REFLECT_INSTRUCTION_GSM8K,
        additional_keys={},
    )
    assert out == gt_out


def test_reflexion_cot_reflect_condition() -> None:
    """Tests ReflexionCoTMathStrategy reflect_condition."""
    llm = FakeListChatModel(responses=[])
    strategy = ReflexionCoTMathStrategy(llm)

    assert not strategy.reflect_condition(0, "strategy1", "key1")
    assert strategy.reflect_condition(1, "strategy1", "key1")
    assert strategy.reflect_condition(1, "strategy1", "key2")
    assert strategy.reflect_condition(1, "", "key2")


def test_reflexion_cot_instantiate_strategies() -> None:
    """Tests ReflexionCoTMathStrategy instantiate strategies."""
    llm = FakeListChatModel(responses=[])
    gsm8k_strategy = ReflexionCoTGSM8KStrategy(llm=llm)
    svamp_strategy = ReflexionCoTSVAMPStrategy(llm=llm)
    tabmwp_strategy = ReflexionCoTTabMWPStrategy(llm=llm)

    assert isinstance(gsm8k_strategy, ReflexionCoTGSM8KStrategy)
    assert isinstance(svamp_strategy, ReflexionCoTSVAMPStrategy)
    assert isinstance(tabmwp_strategy, ReflexionCoTTabMWPStrategy)


def test_reflexion_react_init() -> None:
    """Tests ReflexionReActMathStrategy init."""
    llm = FakeListChatModel(responses=[])
    strategy = ReflexionReActMathStrategy(llm=llm)
    assert isinstance(strategy.llm, BaseChatModel)
    assert isinstance(strategy.reflector, ReflexionReActReflector)
    assert strategy.max_reflections == 3
    assert strategy.max_trials == 1
    assert strategy._scratchpad == ""
    assert strategy._finished == False
    assert strategy._answer == ""


def test_reflexion_react_generate() -> None:
    """Tests ReflexionReActMathStrategy generate."""
    question = "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with 4933828. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"

    gt_scratchpad = "\nThought: I need to calculate how much money Janet makes at the farmers' market daily based on the number of fresh duck eggs she sells."
    gt_out = "I need to calculate how much money Janet makes at the farmers' market daily based on the number of fresh duck eggs she sells."
    responses = [
        "I need to calculate how much money Janet makes at the farmers' market daily based on the number of fresh duck eggs she sells.\n\nAction: Calculate[\n```python\neggs_laid_per_day = 16\neggs_for_breakfast = 3\neggs_baked_into_muffins = 4933828\neggs_sold = eggs_laid_per_day - eggs_for_breakfast - eggs_baked_into_muffins\nprice_per_egg = 2\ntotal_money_daily = eggs_sold * price_per_egg\nanswer = total_money_daily\n```\n]\n\nObservation: \n```python\neggs_laid_per_day = 16\neggs_for_breakfast = 3\neggs_baked_into_muffins = 4933828\neggs_sold = eggs_laid_per_day - eggs_for_breakfast - eggs_baked_into_muffins\nprice_per_egg = 2\ntotal_money_daily = eggs_sold * price_per_egg\nanswer = total_money_daily\n```\nExecution Status: Done\nOutput: answer = -9867653\n\nThought: The answer is negative, which doesn't make sense. I must have made a mistake in the calculation.\n\nAction: Calculate[\n```python\neggs_laid_per_day = 16\neggs_for_breakfast = 3\neggs_baked_into_muffins = 4933828\neggs_sold = eggs_laid_per_day - eggs_for_breakfast - eggs_baked_into_muffins\nprice_per_egg = 2\ntotal_money_daily = eggs_sold * price_per_egg\ntotal_money_daily = abs(total_money_daily)  # Taking the absolute value\nanswer = total_money_daily\n```\n]\n\nObservation: \n```python\neggs_laid_per_day = 16\neggs_for_breakfast = 3\neggs_baked_into_muffins = 4933828\neggs_sold = eggs_laid_per_day - eggs_for_breakfast - eggs_baked_into_muffins\nprice_per_egg = 2\ntotal_money_daily = eggs_sold * price_per_egg\ntotal_money_daily = abs(total_money_daily)  # Taking the absolute value\nanswer = total_money_daily\n```\nExecution Status: Done\nOutput: answer = 9867646\n\nThought: Janet makes $9867646 every day at the farmers' market.\nAction: Finish[\n```python\nanswer = 9867646\n```\n]\n\nObservation: \n```python\nanswer = 9867646\n```"
    ]
    llm = FakeListChatModel(responses=responses)
    strategy = ReflexionReActMathStrategy(llm=llm)
    out = strategy.generate(
        question=question,
        examples=GSM8K_FEWSHOT_EXAMPLES_REACT,
        reflections="",
        prompt=REFLEXION_REACT_INSTRUCTION_GSM8K,
        additional_keys={},
        max_steps=5,
    )
    assert out == gt_out
    assert strategy._scratchpad == gt_scratchpad


def test_reflexion_react_generate_action() -> None:
    """Tests ReflexionReActMathStrategy generate_action."""
    question = "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with 4933828. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"

    gt_scratchpad = "\nAction: Calculate[\n```python\neggs_laid_per_day = 16\neggs_for_breakfast = 3\neggs_used_in_muffins = 4933828\neggs_sold = eggs_laid_per_day - eggs_for_breakfast - eggs_used_in_muffins\nprice_per_egg = 2\ndaily_income = eggs_sold * price_per_egg\nanswer = daily_income\n```\n]"
    responses = [
        "Calculate[\n```python\neggs_laid_per_day = 16\neggs_for_breakfast = 3\neggs_used_in_muffins = 4933828\neggs_sold = eggs_laid_per_day - eggs_for_breakfast - eggs_used_in_muffins\nprice_per_egg = 2\ndaily_income = eggs_sold * price_per_egg\nanswer = daily_income\n```\n]"
    ]
    llm = FakeListChatModel(responses=responses)
    strategy = ReflexionReActMathStrategy(llm=llm)
    action_type, query = strategy.generate_action(
        question=question,
        examples=GSM8K_FEWSHOT_EXAMPLES_REACT,
        reflections="",
        prompt=REFLEXION_REACT_INSTRUCTION_GSM8K,
        additional_keys={},
        max_steps=5,
    )
    assert action_type == "Calculate"
    assert (
        query
        == "eggs_laid_per_day = 16\neggs_for_breakfast = 3\neggs_used_in_muffins = 4933828\neggs_sold = eggs_laid_per_day - eggs_for_breakfast - eggs_used_in_muffins\nprice_per_egg = 2\ndaily_income = eggs_sold * price_per_egg\nanswer = daily_income"
    )
    assert strategy._scratchpad == gt_scratchpad


def test_reflexion_react_generate_observation() -> None:
    """Tests ReflexionReActMathStrategy generate_observation."""
    llm = FakeListChatModel(responses=[])
    strategy = ReflexionReActMathStrategy(llm=llm)

    # Test Calculate.
    is_correct, obs, external_tool_info = strategy.generate_observation(
        step_idx=1,
        action_type="Calculate",
        query="eggs_laid_per_day = 16\neggs_for_breakfast = 3\neggs_used_in_muffins = 4933828\neggs_sold = eggs_laid_per_day - eggs_for_breakfast - eggs_used_in_muffins\nprice_per_egg = 2\ndaily_income = eggs_sold * price_per_egg\nanswer = daily_income",
        key=-9867630,
    )
    assert is_correct
    assert (
        obs
        == "\n```python\neggs_laid_per_day = 16\neggs_for_breakfast = 3\neggs_used_in_muffins = 4933828\neggs_sold = eggs_laid_per_day - eggs_for_breakfast - eggs_used_in_muffins\nprice_per_egg = 2\ndaily_income = eggs_sold * price_per_egg\nanswer = daily_income\n```\nExecution Status: Done\nOutput: answer = -9867630"
    )
    assert external_tool_info == {"execution_status": "Done", "code_answer": -9867630}

    # Test Finish incorrect.
    is_correct, obs, external_tool_info = strategy.generate_observation(
        step_idx=1,
        action_type="Finish",
        query="answer = 5",
        key="key1",
    )
    assert not is_correct
    assert obs == "Answer is INCORRECT"
    assert strategy._scratchpad != ""
    assert strategy._finished
    assert strategy._answer == "answer = 5"
    assert external_tool_info == {"code_answer": 5, "execution_status": "Done"}

    # Test Finish correct.
    is_correct, obs, external_tool_info = strategy.generate_observation(
        step_idx=1,
        action_type="Finish",
        query="answer = 5",
        key=5,
    )
    assert is_correct
    assert obs == "Answer is CORRECT"
    assert strategy._scratchpad != ""
    assert strategy._finished
    assert strategy._answer == "answer = 5"
    assert external_tool_info == {"code_answer": 5, "execution_status": "Done"}

    # Test invalid.
    is_correct, obs, external_tool_info = strategy.generate_observation(
        step_idx=1,
        action_type="Invalid",
        query="answer = 5",
        key=5,
    )
    assert is_correct
    assert (
        obs == "Invalid Action. Valid Actions are Calculate[code] and Finish[answer]."
    )
    assert strategy._scratchpad != ""
    assert strategy._finished
    assert strategy._answer == "answer = 5"
    assert external_tool_info == {"code_answer": "", "execution_status": ""}


def test_reflexion_react_create_output_dict() -> None:
    """Tests ReflexionReActMathStrategy create_output_dict."""
    strategy = ReflexionReActMathStrategy(llm=FakeListChatModel(responses=[]))
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


def test_reflexion_react_react_create_output_dict() -> None:
    """Tests ReflexionReActMathStrategy react_create_output_dict."""
    strategy = ReflexionReActMathStrategy(llm=FakeListChatModel(responses=[]))

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


def test_reflexion_react_halting_condition() -> None:
    """Tests ReflexionReActMathStrategy halting_condition."""
    llm = FakeListChatModel(responses=[])

    # Test case 1: Halting condition met because answer is incorrect and index is less than max_trials.
    strategy = ReflexionReActMathStrategy(llm=llm, max_trials=5)
    strategy._answer = "incorrect_answer"
    assert strategy.halting_condition(3, "correct_answer") == False

    # Test case 2: Halting condition not met because answer is correct.
    strategy = ReflexionReActMathStrategy(llm=llm, max_trials=5)
    strategy._answer = "correct_answer"
    assert strategy.halting_condition(3, "correct_answer") == False

    # Test case 3: Halting condition not met because index is greater than or equal to max_trials.
    strategy = ReflexionReActMathStrategy(llm=llm, max_trials=3)
    strategy._answer = "incorrect_answer"
    assert strategy.halting_condition(4, "correct_answer") == True

    # Test case 4: Halting condition met using max_trials from kwargs.
    strategy = ReflexionReActMathStrategy(llm=llm, max_trials=5)
    strategy._answer = "incorrect_answer"
    assert strategy.halting_condition(3, "correct_answer", max_trials=4) == False

    # Test case 5: Halting condition not met using max_trials from kwargs.
    strategy = ReflexionReActMathStrategy(llm=llm, max_trials=5)
    strategy._answer = "incorrect_answer"
    assert strategy.halting_condition(4, "correct_answer", max_trials=3) == True


def test_reflexion_react_react_halting_condition() -> None:
    """Tests ReflexionReActMathStrategy react_halting_condition."""
    strategy = ReflexionReActMathStrategy(llm=FakeListChatModel(responses=[]))

    idx = 0
    question = "What is the capital of France?"
    examples = ""
    reflections = ""
    prompt = "Answer the question."

    assert not strategy.react_halting_condition(
        idx, question, examples, reflections, prompt, {}
    )


def test_reflexion_react_reset() -> None:
    """Tests ReflexionReActMathStrategy reset."""
    llm = FakeListChatModel(responses=[])
    strategy = ReflexionReActMathStrategy(llm=llm)
    strategy._scratchpad = "Some previous state"
    strategy._finished = True

    strategy.reset()

    assert strategy._scratchpad == ""
    assert not strategy._finished


def test_reflexion_react_reflect() -> None:
    """Tests ReflexionReActMathStrategy reflect."""
    question = "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with 4933828. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"

    gt_reflections = "You have attempted to answer following question before and failed. The following reflection(s) give a plan to avoid failing to answer the question in the same way you did previously. Use them to improve your strategy of correctly answering the given question.\nReflections:\n- 1"
    llm = FakeListChatModel(responses=["1"])
    strategy = ReflexionReActMathStrategy(llm=llm)
    _, reflections = strategy.reflect(
        reflect_strategy="reflexion",
        question=question,
        examples=GSM8K_FEWSHOT_EXAMPLES_REFLEXION_REACT_REFLECT,
        prompt=REFLEXION_REACT_REFLECT_INSTRUCTION_GSM8K,
        additional_keys={},
    )
    assert reflections == gt_reflections


def test_reflexion_react_reflect_condition() -> None:
    """Tests ReflexionReActMathStrategy reflect_condition."""
    question = "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with 4933828. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"

    llm = FakeListChatModel(responses=["1"])
    strategy = ReflexionReActMathStrategy(llm=llm)
    out = strategy.reflect_condition(
        step_idx=1,
        reflect_strategy="reflexion",
        question=question,
        examples=GSM8K_FEWSHOT_EXAMPLES_REFLEXION_REACT_REFLECT,
        key="key",
        prompt=REFLEXION_REACT_REFLECT_INSTRUCTION_GSM8K,
        additional_keys={},
    )
    assert not out


def test_reflexion_react_instantiate_strategies() -> None:
    """Tests ReflexionReActMathStrategy instantiate strategies."""
    llm = FakeListChatModel(responses=[])
    gsm8k_strategy = ReflexionReActMathStrategy(llm=llm)
    svamp_strategy = ReflexionReActSVAMPStrategy(llm=llm)
    tabmwp_strategy = ReflexionReActTabMWPStrategy(llm=llm)

    assert isinstance(gsm8k_strategy, ReflexionReActMathStrategy)
    assert isinstance(svamp_strategy, ReflexionReActSVAMPStrategy)
    assert isinstance(tabmwp_strategy, ReflexionReActTabMWPStrategy)
