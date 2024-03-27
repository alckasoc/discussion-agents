"""Unit tests for ExpeL functional module."""

import joblib

from langchain_community.chat_models.fake import FakeListChatModel
from langchain_core.messages.human import HumanMessage

from discussion_agents.cog.agent.reflexion import ReflexionReActAgent
from discussion_agents.cog.functional.expel import (
    _build_all_success_prompt,
    _build_compare_prompt,
    categorize_experiences,
    create_rules,
    gather_experience,
    get_folds,
    is_existing_rule,
    parse_insights,
    remove_err_operations,
    retrieve_insight_index,
    update_rules,
    get_operations_compare,
    get_operations_success
)


def test_gather_experience(reflexion_react_agent: ReflexionReActAgent) -> None:
    """Test gather_experience."""
    questions = [""]
    keys = [""]
    experiences = gather_experience(
        reflexion_react_agent,
        questions,
        keys,
    )
    gt_experiences = {
        "idxs": [0],
        "questions": [""],
        "keys": [""],
        "trajectories": [[]],
        "reflections": [[]],
    }
    assert experiences == gt_experiences


def test_categorize_experiences(expel_experiences_10_fake_path: str) -> None:
    """Test categorize_experiences."""
    experiences = joblib.load(expel_experiences_10_fake_path)
    categories = categorize_experiences(experiences)
    gt_categories = {"compare": [6, 7, 8, 9], "success": [3, 5], "fail": [0, 1, 2, 4]}
    assert categories == gt_categories


def test_get_folds() -> None:
    """Test get_folds."""
    gt_folds = {0: [1, 4, 5, 8, 11, 14], 1: [0, 2, 3, 6, 7, 9, 10, 12, 13]}

    categories = {
        "compare": [10, 11, 12, 13, 14],
        "success": [1, 3, 6, 7, 8],
        "fail": [0, 2, 4, 5, 9],
    }
    folds = get_folds(categories, n_instances=15)
    assert folds == gt_folds


def test__build_compare_prompt() -> None:
    """Test _build_compare_prompt."""
    # Test is_full=True, empty insights.
    gt_prompt_msgs = [
        HumanMessage(
            content="You are an advanced reasoning agent that can add, edit or remove rules from your existing rule set, based on forming new critiques of past task trajectories. You will be given two previous task trials in which you were given access to a Docstore API environment and a question to answer: one successful and one unsuccessful trial. You failed the trial either because you guessed the wrong answer with Finish[<answer>], or you used up your set number of reasoning steps."
        ),
        HumanMessage(
            content="\nHere are the two previous trials to compare and critique:\nTRIAL TASK:\n\n\nSUCCESSFUL TRIAL:\n\n\nFAILED TRIAL:\n\n\nHere are the EXISTING RULES:\n1. \n\nBy examining and contrasting to the successful trial, and the list of existing rules, you can perform the following operations: add, edit, remove, or agree so that the new list of rules is GENERAL and HIGH LEVEL critiques of the failed trial or proposed way of Thought so they can be used to avoid similar failures when encountered with different questions in the future. Have an emphasis on critiquing how to perform better Thought and Action. Follow the below format:\n\n<OPERATION> <RULE NUMBER>: <RULE>\n\nThe available operations are: AGREE (if the existing rule is strongly relevant for the task), REMOVE (if one existing rule is contradictory or similar/duplicated to other existing rules), EDIT (if any existing rule is not general enough or can be enhanced, rewrite and improve it), ADD (add new rules that are very different from existing rules and relevant for other tasks). Each needs to CLOSELY follow their corresponding formatting below (any existing rule not edited, not agreed, nor removed is considered copied):\n\nAGREE <EXISTING RULE NUMBER>: <EXISTING RULE>\nREMOVE <EXISTING RULE NUMBER>: <EXISTING RULE>\nEDIT <EXISTING RULE NUMBER>: <NEW MODIFIED RULE>\nADD <NEW RULE NUMBER>: <NEW RULE>\n\nDo not mention the trials in the rules because all the rules should be GENERALLY APPLICABLE. Each rule should be concise and easy to follow. Any operation can be used MULTIPLE times. Do at most 4 operations and each existing rule can only get a maximum of 1 operation. Focus on REMOVE rules first, and stop ADD rule unless the new rule is VERY insightful and different from EXISTING RULES. Below are the operations you do to the above list of EXISTING RULES:"
        ),
    ]
    prompt = _build_compare_prompt(
        insights=[], question="", success_trial="", failed_trial="", is_full=True
    )
    assert prompt == "\n".join([p.content for p in gt_prompt_msgs])

    # Test is_full=True, non-empty rules.
    gt_prompt_msgs = [
        HumanMessage(
            content="You are an advanced reasoning agent that can add, edit or remove rules from your existing rule set, based on forming new critiques of past task trajectories. You will be given two previous task trials in which you were given access to a Docstore API environment and a question to answer: one successful and one unsuccessful trial. You failed the trial either because you guessed the wrong answer with Finish[<answer>], or you used up your set number of reasoning steps."
        ),
        HumanMessage(
            content="\nHere are the two previous trials to compare and critique:\nTRIAL TASK:\n\n\nSUCCESSFUL TRIAL:\n\n\nFAILED TRIAL:\n\n\nHere are the EXISTING RULES:\n1. a\n2. b\n\nBy examining and contrasting to the successful trial, and the list of existing rules, you can perform the following operations: add, edit, remove, or agree so that the new list of rules is GENERAL and HIGH LEVEL critiques of the failed trial or proposed way of Thought so they can be used to avoid similar failures when encountered with different questions in the future. Have an emphasis on critiquing how to perform better Thought and Action. Follow the below format:\n\n<OPERATION> <RULE NUMBER>: <RULE>\n\nThe available operations are: AGREE (if the existing rule is strongly relevant for the task), REMOVE (if one existing rule is contradictory or similar/duplicated to other existing rules), EDIT (if any existing rule is not general enough or can be enhanced, rewrite and improve it), ADD (add new rules that are very different from existing rules and relevant for other tasks). Each needs to CLOSELY follow their corresponding formatting below (any existing rule not edited, not agreed, nor removed is considered copied):\n\nAGREE <EXISTING RULE NUMBER>: <EXISTING RULE>\nREMOVE <EXISTING RULE NUMBER>: <EXISTING RULE>\nEDIT <EXISTING RULE NUMBER>: <NEW MODIFIED RULE>\nADD <NEW RULE NUMBER>: <NEW RULE>\n\nDo not mention the trials in the rules because all the rules should be GENERALLY APPLICABLE. Each rule should be concise and easy to follow. Any operation can be used MULTIPLE times. Do at most 4 operations and each existing rule can only get a maximum of 1 operation. Focus on REMOVE rules first, and stop ADD rule unless the new rule is VERY insightful and different from EXISTING RULES. Below are the operations you do to the above list of EXISTING RULES:"
        ),
    ]
    prompt = _build_compare_prompt(
        insights=[("a", 0), ("b", 0)],
        question="",
        success_trial="",
        failed_trial="",
        is_full=True,
    )
    assert prompt == "\n".join([p.content for p in gt_prompt_msgs])

    # Test is_full=False, empty rules.
    gt_prompt_msgs = [
        HumanMessage(
            content="You are an advanced reasoning agent that can add, edit or remove rules from your existing rule set, based on forming new critiques of past task trajectories. You will be given two previous task trials in which you were given access to a Docstore API environment and a question to answer: one successful and one unsuccessful trial. You failed the trial either because you guessed the wrong answer with Finish[<answer>], or you used up your set number of reasoning steps."
        ),
        HumanMessage(
            content="\nHere are the two previous trials to compare and critique:\nTRIAL TASK:\n\n\nSUCCESSFUL TRIAL:\n\n\nFAILED TRIAL:\n\n\nHere are the EXISTING RULES:\n1. \n\nBy examining and contrasting to the successful trial, and the list of existing rules, you can perform the following operations: add, edit, remove, or agree so that the new list of rules is GENERAL and HIGH LEVEL critiques of the failed trial or proposed way of Thought so they can be used to avoid similar failures when encountered with different questions in the future. Have an emphasis on critiquing how to perform better Thought and Action. Follow the below format:\n\n<OPERATION> <RULE NUMBER>: <RULE>\n\nThe available operations are: AGREE (if the existing rule is strongly relevant for the task), REMOVE (if one existing rule is contradictory or similar/duplicated to other existing rules), EDIT (if any existing rule is not general enough or can be enhanced, rewrite and improve it), ADD (add new rules that are very different from existing rules and relevant for other tasks). Each needs to CLOSELY follow their corresponding formatting below (any existing rule not edited, not agreed, nor removed is considered copied):\n\nAGREE <EXISTING RULE NUMBER>: <EXISTING RULE>\nREMOVE <EXISTING RULE NUMBER>: <EXISTING RULE>\nEDIT <EXISTING RULE NUMBER>: <NEW MODIFIED RULE>\nADD <NEW RULE NUMBER>: <NEW RULE>\n\nDo not mention the trials in the rules because all the rules should be GENERALLY APPLICABLE. Each rule should be concise and easy to follow. Any operation can be used MULTIPLE times. Do at most 4 operations and each existing rule can only get a maximum of 1 operation. Below are the operations you do to the above list of EXISTING RULES:"
        ),
    ]
    prompt = _build_compare_prompt(
        insights=[], question="", success_trial="", failed_trial="", is_full=False
    )
    assert prompt == "\n".join([p.content for p in gt_prompt_msgs])

    # Test is_full=False, non-empty rules.
    gt_prompt_msgs = [
        HumanMessage(
            content="You are an advanced reasoning agent that can add, edit or remove rules from your existing rule set, based on forming new critiques of past task trajectories. You will be given two previous task trials in which you were given access to a Docstore API environment and a question to answer: one successful and one unsuccessful trial. You failed the trial either because you guessed the wrong answer with Finish[<answer>], or you used up your set number of reasoning steps."
        ),
        HumanMessage(
            content="\nHere are the two previous trials to compare and critique:\nTRIAL TASK:\n\n\nSUCCESSFUL TRIAL:\n\n\nFAILED TRIAL:\n\n\nHere are the EXISTING RULES:\n1. a\n2. b\n\nBy examining and contrasting to the successful trial, and the list of existing rules, you can perform the following operations: add, edit, remove, or agree so that the new list of rules is GENERAL and HIGH LEVEL critiques of the failed trial or proposed way of Thought so they can be used to avoid similar failures when encountered with different questions in the future. Have an emphasis on critiquing how to perform better Thought and Action. Follow the below format:\n\n<OPERATION> <RULE NUMBER>: <RULE>\n\nThe available operations are: AGREE (if the existing rule is strongly relevant for the task), REMOVE (if one existing rule is contradictory or similar/duplicated to other existing rules), EDIT (if any existing rule is not general enough or can be enhanced, rewrite and improve it), ADD (add new rules that are very different from existing rules and relevant for other tasks). Each needs to CLOSELY follow their corresponding formatting below (any existing rule not edited, not agreed, nor removed is considered copied):\n\nAGREE <EXISTING RULE NUMBER>: <EXISTING RULE>\nREMOVE <EXISTING RULE NUMBER>: <EXISTING RULE>\nEDIT <EXISTING RULE NUMBER>: <NEW MODIFIED RULE>\nADD <NEW RULE NUMBER>: <NEW RULE>\n\nDo not mention the trials in the rules because all the rules should be GENERALLY APPLICABLE. Each rule should be concise and easy to follow. Any operation can be used MULTIPLE times. Do at most 4 operations and each existing rule can only get a maximum of 1 operation. Below are the operations you do to the above list of EXISTING RULES:"
        ),
    ]
    prompt = _build_compare_prompt(
        insights=[("a", 0), ("b", 0)],
        question="",
        success_trial="",
        failed_trial="",
        is_full=False,
    )
    assert prompt == "\n".join([p.content for p in gt_prompt_msgs])


def test__build_all_success_prompt() -> None:
    """Test _build_all_success_prompt."""
    # Test is_full=True, empty rules.
    gt_prompt_msgs = [
        HumanMessage(
            content="You are an advanced reasoning agent that can add, edit or remove rules from your existing rule set, based on forming new critiques of past task trajectories. You will be given successful tasks trials in which you were given access to a Docstore API environment and a question to answer."
        ),
        HumanMessage(
            content="\nHere are the trials:\n\n\nHere are the EXISTING RULES:\n1. \n\nBy examining the successful trials, and the list of existing rules, you can perform the following operations: add, edit, remove, or agree so that the new list of rules are general and high level insights of the successful trials or proposed way of Thought so they can be used as helpful tips to different tasks in the future. Have an emphasis on tips that help the agent perform better Thought and Action. Follow the below format:\n\n<OPERATION> <RULE NUMBER>: <RULE>\n\nThe available operations are: AGREE (if the existing rule is strongly relevant for the task), REMOVE (if one existing rule is contradictory or similar/duplicated to other existing rules), EDIT (if any existing rule is not general enough or can be enhanced, rewrite and improve it), ADD (add new rules that are very different from existing rules and relevant for other tasks). Each needs to CLOSELY follow their corresponding formatting below (any existing rule not edited, not agreed, nor removed is considered copied):\n\nAGREE <EXISTING RULE NUMBER>: <EXISTING RULE>\nREMOVE <EXISTING RULE NUMBER>: <EXISTING RULE>\nEDIT <EXISTING RULE NUMBER>: <NEW MODIFIED RULE>\nADD <NEW RULE NUMBER>: <NEW RULE>\n\nDo not mention the trials in the rules because all the rules should be GENERALLY APPLICABLE. Each rule should be concise and easy to follow. Any operation can be used MULTIPLE times. Do at most 4 operations and each existing rule can only get a maximum of 1 operation. Focus on REMOVE rules first, and stop ADD rule unless the new rule is VERY insightful and different from EXISTING RULES. Below are the operations you do to the above list of EXISTING RULES:"
        ),
    ]
    prompt = _build_all_success_prompt(insights=[], success_trajs_str="", is_full=True)
    assert prompt == "\n".join([p.content for p in gt_prompt_msgs])

    # Test is_full=True, non-empty rules.
    gt_prompt_msgs = [
        HumanMessage(
            content="You are an advanced reasoning agent that can add, edit or remove rules from your existing rule set, based on forming new critiques of past task trajectories. You will be given successful tasks trials in which you were given access to a Docstore API environment and a question to answer."
        ),
        HumanMessage(
            content="\nHere are the trials:\n\n\nHere are the EXISTING RULES:\n1. a\n2. b\n\nBy examining the successful trials, and the list of existing rules, you can perform the following operations: add, edit, remove, or agree so that the new list of rules are general and high level insights of the successful trials or proposed way of Thought so they can be used as helpful tips to different tasks in the future. Have an emphasis on tips that help the agent perform better Thought and Action. Follow the below format:\n\n<OPERATION> <RULE NUMBER>: <RULE>\n\nThe available operations are: AGREE (if the existing rule is strongly relevant for the task), REMOVE (if one existing rule is contradictory or similar/duplicated to other existing rules), EDIT (if any existing rule is not general enough or can be enhanced, rewrite and improve it), ADD (add new rules that are very different from existing rules and relevant for other tasks). Each needs to CLOSELY follow their corresponding formatting below (any existing rule not edited, not agreed, nor removed is considered copied):\n\nAGREE <EXISTING RULE NUMBER>: <EXISTING RULE>\nREMOVE <EXISTING RULE NUMBER>: <EXISTING RULE>\nEDIT <EXISTING RULE NUMBER>: <NEW MODIFIED RULE>\nADD <NEW RULE NUMBER>: <NEW RULE>\n\nDo not mention the trials in the rules because all the rules should be GENERALLY APPLICABLE. Each rule should be concise and easy to follow. Any operation can be used MULTIPLE times. Do at most 4 operations and each existing rule can only get a maximum of 1 operation. Focus on REMOVE rules first, and stop ADD rule unless the new rule is VERY insightful and different from EXISTING RULES. Below are the operations you do to the above list of EXISTING RULES:"
        ),
    ]
    prompt = _build_all_success_prompt(
        insights=[("a", 0), ("b", 0)], success_trajs_str="", is_full=True
    )
    assert prompt == "\n".join([p.content for p in gt_prompt_msgs])

    # Test is_full=False, empty rules.
    gt_prompt_msgs = [
        HumanMessage(
            content="You are an advanced reasoning agent that can add, edit or remove rules from your existing rule set, based on forming new critiques of past task trajectories. You will be given successful tasks trials in which you were given access to a Docstore API environment and a question to answer."
        ),
        HumanMessage(
            content="\nHere are the trials:\n\n\nHere are the EXISTING RULES:\n1. \n\nBy examining the successful trials, and the list of existing rules, you can perform the following operations: add, edit, remove, or agree so that the new list of rules are general and high level insights of the successful trials or proposed way of Thought so they can be used as helpful tips to different tasks in the future. Have an emphasis on tips that help the agent perform better Thought and Action. Follow the below format:\n\n<OPERATION> <RULE NUMBER>: <RULE>\n\nThe available operations are: AGREE (if the existing rule is strongly relevant for the task), REMOVE (if one existing rule is contradictory or similar/duplicated to other existing rules), EDIT (if any existing rule is not general enough or can be enhanced, rewrite and improve it), ADD (add new rules that are very different from existing rules and relevant for other tasks). Each needs to CLOSELY follow their corresponding formatting below (any existing rule not edited, not agreed, nor removed is considered copied):\n\nAGREE <EXISTING RULE NUMBER>: <EXISTING RULE>\nREMOVE <EXISTING RULE NUMBER>: <EXISTING RULE>\nEDIT <EXISTING RULE NUMBER>: <NEW MODIFIED RULE>\nADD <NEW RULE NUMBER>: <NEW RULE>\n\nDo not mention the trials in the rules because all the rules should be GENERALLY APPLICABLE. Each rule should be concise and easy to follow. Any operation can be used MULTIPLE times. Do at most 4 operations and each existing rule can only get a maximum of 1 operation. Below are the operations you do to the above list of EXISTING RULES:"
        ),
    ]
    prompt = _build_all_success_prompt(insights=[], success_trajs_str="", is_full=False)
    assert prompt == "\n".join([p.content for p in gt_prompt_msgs])

    # Test is_full=False, non-empty rules.
    gt_prompt_msgs = [
        HumanMessage(
            content="You are an advanced reasoning agent that can add, edit or remove rules from your existing rule set, based on forming new critiques of past task trajectories. You will be given successful tasks trials in which you were given access to a Docstore API environment and a question to answer."
        ),
        HumanMessage(
            content="\nHere are the trials:\n\n\nHere are the EXISTING RULES:\n1. a\n2. b\n\nBy examining the successful trials, and the list of existing rules, you can perform the following operations: add, edit, remove, or agree so that the new list of rules are general and high level insights of the successful trials or proposed way of Thought so they can be used as helpful tips to different tasks in the future. Have an emphasis on tips that help the agent perform better Thought and Action. Follow the below format:\n\n<OPERATION> <RULE NUMBER>: <RULE>\n\nThe available operations are: AGREE (if the existing rule is strongly relevant for the task), REMOVE (if one existing rule is contradictory or similar/duplicated to other existing rules), EDIT (if any existing rule is not general enough or can be enhanced, rewrite and improve it), ADD (add new rules that are very different from existing rules and relevant for other tasks). Each needs to CLOSELY follow their corresponding formatting below (any existing rule not edited, not agreed, nor removed is considered copied):\n\nAGREE <EXISTING RULE NUMBER>: <EXISTING RULE>\nREMOVE <EXISTING RULE NUMBER>: <EXISTING RULE>\nEDIT <EXISTING RULE NUMBER>: <NEW MODIFIED RULE>\nADD <NEW RULE NUMBER>: <NEW RULE>\n\nDo not mention the trials in the rules because all the rules should be GENERALLY APPLICABLE. Each rule should be concise and easy to follow. Any operation can be used MULTIPLE times. Do at most 4 operations and each existing rule can only get a maximum of 1 operation. Below are the operations you do to the above list of EXISTING RULES:"
        ),
    ]
    prompt = _build_all_success_prompt(
        insights=[("a", 0), ("b", 0)], success_trajs_str="", is_full=False
    )
    assert prompt == "\n".join([p.content for p in gt_prompt_msgs])


def test_parse_insights() -> None:
    """Test parse_insights."""
    gt_rules = [
        ("REMOVE 1", "Rule to remove."),
        ("EDIT 2", "Rule to edit."),
        ("ADD", "Rule to add."),
    ]
    llm_output = "REMOVE 1: Rule to remove.\nEDIT 2: Rule to edit.\nADD 3: Rule to add."
    rules = parse_insights(llm_output)
    assert rules == gt_rules

    gt_rules = [("AGREE 1", "This is a valid rule.")]
    llm_output = "AGREE 1: This is a valid rule."
    rules = parse_insights(llm_output)
    assert rules == gt_rules


def test_retrieve_insight_index() -> None:
    """Tests retrieve_insight_index."""
    rules = [("Rule1", 1), ("Rule2", 2), ("Rule3", 3)]

    idx = retrieve_insight_index(rules, "Operation on Rule1")
    assert idx == 0

    idx = retrieve_insight_index(rules, "Changes to Rule2")
    assert idx == 1

    idx = retrieve_insight_index(rules, "Modification of Rule3")
    assert idx == 2

    idx = retrieve_insight_index(rules, "No such rule")
    assert idx == -1


def test_is_existing_rule() -> None:
    """Tests is_existing_rule."""
    rules = [("Rule1", 1), ("Rule2", 2), ("Rule3", 3)]

    assert is_existing_rule(rules, "Operation on Rule1")
    assert is_existing_rule(rules, "Changes to Rule2")
    assert is_existing_rule(rules, "Modification of Rule3")
    assert not is_existing_rule(rules, "No such rule")


def test_remove_err_operations() -> None:
    """Test remove_err_operations."""
    rules = [("Rule1", 1), ("Rule2", 2)]

    operations = [
        ("ADD 1", "Rule1"),
        ("ADD 2", "Rule3"),
        ("EDIT 1", "Rule1"),
        ("EDIT 3", "Rule3"),
        ("REMOVE", "Rule1"),
        ("REMOVE", "Rule3"),
        ("AGREE", "Rule1"),
        ("AGREE", "Rule3"),
    ]

    expected_operations = [
        ("ADD 2", "Rule3"),
        ("AGREE 1", "Rule1"),
        ("REMOVE", "Rule1"),
        ("AGREE", "Rule1"),
    ]

    out = remove_err_operations(rules, operations)
    assert out == expected_operations


def test_update_rules() -> None:
    """Test update_rules."""
    initial_rules = [("Rule1", 1), ("Rule2", 2), ("Rule3", 3)]
    operations = [
        ("REMOVE", "Rule1"),
        ("AGREE", "Rule2"),
        ("EDIT 3", "Rule3"),
        ("ADD", "NewRule4"),
    ]

    # Expected outcomes.
    expected_rules_full = [("Rule3", 4), ("Rule2", 3), ("NewRule4", 2)]
    expected_rules_not_full = [("Rule3", 4), ("Rule2", 3), ("NewRule4", 2)]

    # Test with is_full=True.
    updated_rules_full = update_rules(initial_rules, operations, is_full=True)
    assert updated_rules_full == expected_rules_full

    # Test with is_full=False.
    updated_rules_not_full = update_rules(initial_rules, operations, is_full=False)
    assert updated_rules_not_full == expected_rules_not_full


def test_get_operations_compare() -> None:
    """Test get_operations_compare."""

    gt_operations = [
        ('ADD', 'When unable to find specific information, consider looking for related topics or broader context that may lead to the desired answer.')
    ]
    insights = []
    question = "VIVA Media AG changed it's name in 2004. What does their new acronym stand for?"
    success_trial = 'Thought: I need to search for the creator of "Wallace and Gromit" and find out what other animation comedy they created that involves animated zoo animals and people talking about their homes. \nAction: Action 1: Search[creator of Wallace and Gromit]\nObservation 1: Invalid Action. Valid Actions are Lookup[<topic>] Search[<topic>] and Finish[<answer>].\nThought: I should search for the creator of "Wallace and Gromit" by searching for Nick Park. \nAction: Search[Nick Park]\nObservation 2: Could not find [Nick Park]. Similar: [\'Nick Park\', \'Wallace and Gromit\', \'Nick Hornby\', \'Aardman Animations\', \'Chicken Run\', \'A Grand Day Out\', \'Harry Gregson-Williams\', \'Plasticine\', \'Nickelodeon\', \'Wallace & Gromit: The Curse of the Were-Rabbit\']\nThought: I need to search for Nick Park using the search term "Nick Park" instead of just Nick Park. \nAction: Search[Nick Park]\nObservation 3: Could not find [Nick Park]. Similar: [\'Nick Park\', \'Wallace and Gromit\', \'Nick Hornby\', \'Aardman Animations\', \'Chicken Run\', \'A Grand Day Out\', \'Harry Gregson-Williams\', \'Plasticine\', \'Nickelodeon\', \'Wallace & Gromit: The Curse of the Were-Rabbit\']\nThought: I should try searching for the creator of "Wallace and Gromit" on a different platform like a search engine. \nAction: Search[creator of Wallace and Gromit]\nObservation 4: Wallace and Gromit is a British stop-motion animated comedy franchise created by Nick Park and produced by Aardman Animations. The main film series consists of four short films and one feature-length film, and has spawned numerous spin-offs and TV adaptations. The series centres on Wallace, a good-natured, eccentric, cheese-loving inventor, and Gromit, his loyal and intelligent anthropomorphic beagle. The first short film, A Grand Day Out, was finished and made public in 1989. Wallace was voiced by actor Peter Sallis until 2010 when he was succeeded by Ben Whitehead. Gromit is largely silent and has no dialogue, communicating through facial expressions and body language.Because of their widespread popularity, the characters have been described as positive international cultural icons of both modern British culture and British people in general. BBC News called them "some of the best-known and best-loved stars to come out of the UK". Icons has said they have done "more to improve the image of the English world-wide than any officially appointed ambassadors". Although not overtly set in any particular town, Park has hinted that it was inspired by 1950s Wigan in Northern England. Wallace\'s accent comes from the Holme Valley of West Yorkshire. Wallace is fond of Wensleydale cheese (from Wensleydale, North Yorkshire).Their films have received universal acclaim, with the first three short films, A Grand Day Out (1989), The Wrong Trousers (1993) and A Close Shave (1995) earning 100% on Rotten Tomatoes; the feature film Wallace & Gromit: The Curse of the Were-Rabbit (2005) has also received acclaim. The feature film is the second-highest-grossing stop-motion animated film, only outgrossed by Chicken Run (2000), another creation of Park\'s. A fourth short film, A Matter of Loaf and Death, was released in 2008. The franchise has received numerous accolades, including five British Academy Film Awards, three Academy Awards and a Peabody Award.The Wallace and Gromit characters spearhead the fundraising for two children\'s charities: Wallace & Gromit\'s Children\'s Foundation, which supports children\'s hospices and hospitals in the United Kingdom, and Wallace and Gromit\'s Grand Appeal, the charity for Bristol Children\'s Hospital in Bristol. In December 2010, Wallace and Gromit featured on a festive themed series of UK postage stamps issued by the Royal Mail.\nThought: The creator of "Wallace and Gromit" is Nick Park. Now, I need to find out what other animation comedy involving animated zoo animals and people talking about their homes was created by him.\nAction: Search[animation comedy animated zoo animals talking about homes Nick Park]\nObservation 5: Nicholas Wulstan Park  (born 6 December 1958) is an  English filmmaker and animator who created Wallace and Gromit, Creature Comforts, Chicken Run, Shaun the Sheep, and Early Man. Park has been nominated for an Academy Award a total of six times and won four with Creature Comforts (1989), The Wrong Trousers (1993), A Close Shave (1995) and Wallace & Gromit: The Curse of the Were-Rabbit (2005).He has also received five BAFTA Awards, including the BAFTA for Best Short Animation for A Matter of Loaf and Death, which was also the most watched television programme in the United Kingdom in 2008. His 2000 film Chicken Run is the highest-grossing stop motion animated film.In 1985, Park joined Aardman Animations based in Bristol, and for his work in animation he was among the British cultural icons selected by artist Peter Blake to appear in a 2012 version of Blake\'s most famous artwork—the Beatles\' Sgt. Pepper\'s Lonely Hearts Club Band album cover—to celebrate the British cultural figures of his life.Park was appointed a Commander of the Order of the British Empire (CBE) in the 1997 Birthday Honours for services to the animated film industry.\nThought: Nick Park created the animation comedy "Creature Comforts" that matched animated zoo animals with a soundtrack of people talking about their homes. \nAction: Finish[Creature Comforts]\nObservation 6: Answer is CORRECT'
    failed_trial = 'Thought: I need to search for VIVA Media AG and find out what their new acronym stands for after changing their name in 2004. \nAction: Search[VIVA Media AG]\nObservation 1: Could not find [VIVA Media AG]. Similar: [\'MTV Music (Polish TV channel)\', \'VIVA Plus\', \'Paramount International Networks\', \'VIVA (German TV channel)\', \'Viacom (1952–2005)\', \'Vauxhall Viva\', \'Sartorius AG\', \'GfK Entertainment charts\', \'Lindt\', \'Puma (brand)\']\nThought: I couldn\'t find VIVA Media AG, so I should try searching for VIVA (German TV channel) instead, as it might be related.\nAction: Search[VIVA (German TV channel)]\nObservation 2: VIVA (abbreviation for: Videoverwertungsanstalt translating to "Video Exploitation Facility") was a German free-to-air music television channel, first broadcast on December 1, 1993. The channel was intended to compete against MTV Europe and was the first German-language music TV channel, while MTV was only broadcast in English until the introduction of MTV Germany in 1997. It was also supposed to focus more on German music and pop culture while MTV only broadcast anglophone music by artists mostly from North America, the United Kingdom, Ireland and Australia.After years of competition for audience share, MTV Networks Europe eventually acquired VIVA on  January 14, 2005, after it had outrun its own efforts for better ratings. MTV operated VIVA channels in Austria, Germany, Hungary, Ireland, Poland, Switzerland and the United Kingdom.In June 2018, Viacom announced that it would shut down all Viva operations worldwide at the end of 2018.The channel ceased to broadcast at 14:00 on 31 December 2018, being replaced by Comedy Central.\nThought: The information provided does not mention any name change or acronym for VIVA Media AG. I should try searching for the specific name change or acronym directly. \nAction: Search[VIVA Media AG name change acronym]\nObservation 3: Could not find [VIVA Media AG name change acronym]. Similar: [\'Holden Torana\', \'Andrés Manuel López Obrador\']\nThought: I need to refine my search and try looking for the specific name change of VIVA Media AG in 2004 to find out what their new acronym stands for. \nAction: Search[VIVA Media AG name change 2004]\nObservation 4: Could not find [VIVA Media AG name change 2004]. Similar: [\'Vauxhall Viva\', \'GfK Entertainment charts\', \'Puma (brand)\', \'Opel Astra\', \'Mediaspree\', \'About You Now\', \'Daihatsu Mira\', \'Bosch (company)\', \'Schneider Electric\', \'Sildenafil\']\nThought: I need to try a different approach. Let me search for the history of VIVA Media AG to see if I can find any information about their name change in 2004 and the new acronym.\nAction: Search[history of VIVA Media AG]\nObservation 5: Could not find [history of VIVA Media AG]. Similar: [\'MTV Music (Polish TV channel)\', \'VIVA Plus\', \'Vauxhall Viva\', \'Sartorius AG\', \'VIVA (German TV channel)\', \'GfK Entertainment charts\', \'Lindt\', \'Spellbound Entertainment\', \'Ag-gag\', \'Puma (brand)\']\nThought: I have exhausted my search options and could not find the information I was looking for. Perhaps I can try a different search engine or look for official company announcements to find out the new acronym for VIVA Media AG after their name change in 2004.\nAction: Finish[unable to find information]\nObservation 6: Answer is INCORRECT'
    is_full = False
    responses = [
        'ADD 11: When unable to find specific information, consider looking for related topics or broader context that may lead to the desired answer.\nREMOVE 3: Search for specific terms in the API environment should include relevant keywords to ensure accurate results.\nEDIT 5: Refine search queries by including additional relevant keywords or context to increase the chances of finding the desired information.\nAGREE 7: Use alternative search engines or sources of information if initial search attempts are unsuccessful.' 
    ]
    operations = get_operations_compare(
        FakeListChatModel(responses=responses),
        insights,
        question,
        success_trial,
        failed_trial,
        is_full
    )
    assert operations == gt_operations


def test_get_operations_success() -> None:
    """Test get_operations success."""

    gt_operations = [
        ('ADD', 'When searching for specific information, try using the full name or specific search terms related to the topic to increase the chances of finding the desired answer.')
    ]
    insights = [{'insight': 'When unable to find specific information, consider looking for related topics or broader context that may lead to the desired answer.', 'score': 2}, {'insight': 'Consider using different search terms or variations when searching for specific information to increase the chances of finding the desired answer.', 'score': 2}]
    success_trials = 'The creator of "Wallace and Gromit" also created what animation comedy that matched animated zoo animals with a soundtrack of people talking about their homes? \nThought: I need to search for the creator of "Wallace and Gromit" and find out what other animation comedy they created that involves animated zoo animals and people talking about their homes. \nAction: Action 1: Search[creator of Wallace and Gromit]\nObservation 1: Invalid Action. Valid Actions are Lookup[<topic>] Search[<topic>] and Finish[<answer>].\nThought: I should search for the creator of "Wallace and Gromit" by searching for Nick Park. \nAction: Search[Nick Park]\nObservation 2: Could not find [Nick Park]. Similar: [\'Nick Park\', \'Wallace and Gromit\', \'Nick Hornby\', \'Aardman Animations\', \'Chicken Run\', \'A Grand Day Out\', \'Harry Gregson-Williams\', \'Plasticine\', \'Nickelodeon\', \'Wallace & Gromit: The Curse of the Were-Rabbit\']\nThought: I need to search for Nick Park using the search term "Nick Park" instead of just Nick Park. \nAction: Search[Nick Park]\nObservation 3: Could not find [Nick Park]. Similar: [\'Nick Park\', \'Wallace and Gromit\', \'Nick Hornby\', \'Aardman Animations\', \'Chicken Run\', \'A Grand Day Out\', \'Harry Gregson-Williams\', \'Plasticine\', \'Nickelodeon\', \'Wallace & Gromit: The Curse of the Were-Rabbit\']\nThought: I should try searching for the creator of "Wallace and Gromit" on a different platform like a search engine. \nAction: Search[creator of Wallace and Gromit]\nObservation 4: Wallace and Gromit is a British stop-motion animated comedy franchise created by Nick Park and produced by Aardman Animations. The main film series consists of four short films and one feature-length film, and has spawned numerous spin-offs and TV adaptations. The series centres on Wallace, a good-natured, eccentric, cheese-loving inventor, and Gromit, his loyal and intelligent anthropomorphic beagle. The first short film, A Grand Day Out, was finished and made public in 1989. Wallace was voiced by actor Peter Sallis until 2010 when he was succeeded by Ben Whitehead. Gromit is largely silent and has no dialogue, communicating through facial expressions and body language.Because of their widespread popularity, the characters have been described as positive international cultural icons of both modern British culture and British people in general. BBC News called them "some of the best-known and best-loved stars to come out of the UK". Icons has said they have done "more to improve the image of the English world-wide than any officially appointed ambassadors". Although not overtly set in any particular town, Park has hinted that it was inspired by 1950s Wigan in Northern England. Wallace\'s accent comes from the Holme Valley of West Yorkshire. Wallace is fond of Wensleydale cheese (from Wensleydale, North Yorkshire).Their films have received universal acclaim, with the first three short films, A Grand Day Out (1989), The Wrong Trousers (1993) and A Close Shave (1995) earning 100% on Rotten Tomatoes; the feature film Wallace & Gromit: The Curse of the Were-Rabbit (2005) has also received acclaim. The feature film is the second-highest-grossing stop-motion animated film, only outgrossed by Chicken Run (2000), another creation of Park\'s. A fourth short film, A Matter of Loaf and Death, was released in 2008. The franchise has received numerous accolades, including five British Academy Film Awards, three Academy Awards and a Peabody Award.The Wallace and Gromit characters spearhead the fundraising for two children\'s charities: Wallace & Gromit\'s Children\'s Foundation, which supports children\'s hospices and hospitals in the United Kingdom, and Wallace and Gromit\'s Grand Appeal, the charity for Bristol Children\'s Hospital in Bristol. In December 2010, Wallace and Gromit featured on a festive themed series of UK postage stamps issued by the Royal Mail.\nThought: The creator of "Wallace and Gromit" is Nick Park. Now, I need to find out what other animation comedy involving animated zoo animals and people talking about their homes was created by him.\nAction: Search[animation comedy animated zoo animals talking about homes Nick Park]\nObservation 5: Nicholas Wulstan Park  (born 6 December 1958) is an  English filmmaker and animator who created Wallace and Gromit, Creature Comforts, Chicken Run, Shaun the Sheep, and Early Man. Park has been nominated for an Academy Award a total of six times and won four with Creature Comforts (1989), The Wrong Trousers (1993), A Close Shave (1995) and Wallace & Gromit: The Curse of the Were-Rabbit (2005).He has also received five BAFTA Awards, including the BAFTA for Best Short Animation for A Matter of Loaf and Death, which was also the most watched television programme in the United Kingdom in 2008. His 2000 film Chicken Run is the highest-grossing stop motion animated film.In 1985, Park joined Aardman Animations based in Bristol, and for his work in animation he was among the British cultural icons selected by artist Peter Blake to appear in a 2012 version of Blake\'s most famous artwork—the Beatles\' Sgt. Pepper\'s Lonely Hearts Club Band album cover—to celebrate the British cultural figures of his life.Park was appointed a Commander of the Order of the British Empire (CBE) in the 1997 Birthday Honours for services to the animated film industry.\nThought: Nick Park created the animation comedy "Creature Comforts" that matched animated zoo animals with a soundtrack of people talking about their homes. \nAction: Finish[Creature Comforts]\nObservation 6: Answer is CORRECT'
    is_full = False
    responses = [
        'ADD 4: When searching for specific information, try using the full name or specific search terms related to the topic to increase the chances of finding the desired answer.' 
    ]
    operations = get_operations_success(
        FakeListChatModel(responses=responses),
        success_trials,
        insights,
        is_full
    )
    assert operations == gt_operations


def test_create_rules(expel_experiences_10_fake_path: str) -> None:
    """Test create_rules."""
    gt_rules = [
        ("Prioritize specific keywords in the question to guide search queries.", 2),
        (
            "Consider alternative search terms if the initial search query does not yield relevant results.",
            2,
        ),
        (
            "Break down complex search queries into smaller, more specific parts to guide the search process effectively.",
            2,
        ),
        (
            "Prioritize refining the search query based on the specific elements of the question to avoid ambiguity and ensure relevance in search results.",
            2,
        ),
        (
            "Prioritize verifying the accuracy of information obtained from search results before providing an answer.",
            2,
        ),
    ]

    max_num_rules = 20
    experiences = joblib.load(expel_experiences_10_fake_path)
    categories = categorize_experiences(experiences)
    folds = get_folds(categories, len(experiences["idxs"]))
    responses = [
        "ADD 1: Prioritize specific keywords in the question to guide search queries.\nEDIT 2: Specify the need to directly find the information requested in the question.\nREMOVE 3: The action is unclear and redundant with previous searches.\nAGREE 4: The action of narrowing down the search to find specific information is valid.",
        "ADD 2: Consider alternative search terms if the initial search query does not yield relevant results.",
        "ADD 3: Break down complex search queries into smaller, more specific parts to guide the search process effectively.",
        "ADD 4: Prioritize refining the search query based on the specific elements of the question to avoid ambiguity and ensure relevance in search results.",
        "ADD 5: Prioritize verifying the accuracy of information obtained from search results before providing an answer.",
    ]
    llm = FakeListChatModel(responses=responses)

    train_idxs = folds[0]
    rules = []
    rules = create_rules(llm, experiences, categories, train_idxs, rules, max_num_rules)
    assert rules == gt_rules
