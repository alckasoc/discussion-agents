"""Unit tests for ExpeL."""

import joblib

from langchain_community.chat_models.fake import FakeListChatModel
from langchain_core.language_models.chat_models import BaseChatModel

from agential.cog.agent.expel import ExpeLAgent
from agential.cog.agent.reflexion import ReflexionReActAgent
from agential.cog.modules.memory.expel import (
    ExpeLExperienceMemory,
    ExpeLInsightMemory,
)


def test_init(expel_experiences_10_fake_path: str) -> None:
    """Test initialization."""
    llm = FakeListChatModel(responses=[])

    agent = ExpeLAgent(llm=llm, mode={"qa": "hotpotqa"})
    assert isinstance(agent.llm, BaseChatModel)
    assert isinstance(agent.reflexion_react_agent, ReflexionReActAgent)
    assert isinstance(agent.experience_memory, ExpeLExperienceMemory)
    assert isinstance(agent.insight_memory, ExpeLInsightMemory)
    assert agent.success_batch_size == 8
    assert agent.experience_memory.experiences == {
        "idxs": [],
        "questions": [],
        "keys": [],
        "trajectories": [],
        "reflections": [],
    }
    assert not agent.experience_memory.success_traj_docs
    assert not agent.experience_memory.vectorstore
    assert not agent.insight_memory.insights

    # Test with all parameters specified except experience memory and reflexion_react_agent.
    agent = ExpeLAgent(
        llm=llm,
        mode={"qa": "hotpotqa"},
        reflexion_react_strategy_kwargs={"max_steps": 3},
        insight_memory=ExpeLInsightMemory(
            insights=[{"insight": "blah blah", "score": 10}]
        ),
        success_batch_size=10,
    )
    assert isinstance(agent.llm, BaseChatModel)
    assert isinstance(agent.reflexion_react_agent, ReflexionReActAgent)
    assert isinstance(agent.experience_memory, ExpeLExperienceMemory)
    assert isinstance(agent.insight_memory, ExpeLInsightMemory)
    assert agent.success_batch_size == 10
    assert agent.experience_memory.experiences == {
        "idxs": [],
        "questions": [],
        "keys": [],
        "trajectories": [],
        "reflections": [],
    }
    assert not agent.experience_memory.success_traj_docs
    assert not agent.experience_memory.vectorstore
    assert agent.insight_memory.insights == [{"insight": "blah blah", "score": 10}]

    # Test with custom reflexion_react_agent (verify it overrides reflexion_react_kwargs)
    agent = ExpeLAgent(
        llm=llm,
        reflexion_react_strategy_kwargs={"max_steps": 100},
        reflexion_react_agent=ReflexionReActAgent(llm=llm, mode={"qa": "hotpotqa"}),
    )
    assert isinstance(agent.reflexion_react_agent, ReflexionReActAgent)
    assert agent.reflexion_react_agent.mode == {"qa": "hotpotqa"}

    # Test with custom experience memory (verify correct initialization).
    experiences = joblib.load(expel_experiences_10_fake_path)
    experiences = {key: value[:1] for key, value in experiences.items()}

    agent = ExpeLAgent(
        llm=llm,
        mode={"qa": "hotpotqa"},
        experience_memory=ExpeLExperienceMemory(experiences),
    )
    assert agent.experience_memory.experiences == experiences
    assert agent.insight_memory.insights == []


def test_reset() -> None:
    """Test reset."""
    llm = FakeListChatModel(responses=["1"])

    agent = ExpeLAgent(llm=llm, mode={"qa": "hotpotqa"})
    agent.reflexion_react_agent.strategy._scratchpad == "cat"
    agent.experience_memory.experiences == "dog"
    agent.insight_memory.insights = ["turtle"]
    agent.reset()
    assert agent.reflexion_react_agent.strategy._scratchpad == ""
    assert agent.experience_memory.experiences == {
        "idxs": [],
        "questions": [],
        "keys": [],
        "trajectories": [],
        "reflections": [],
    }
    assert agent.insight_memory.insights == []


def test_retrieve() -> None:
    """Test retrieve."""
    llm = FakeListChatModel(responses=["1"])

    agent = ExpeLAgent(llm=llm, mode={"qa": "hotpotqa"})
    memory = agent.retrieve()
    assert list(memory.keys()) == [
        "experiences",
        "success_traj_docs",
        "vectorstore",
        "insights",
    ]
    assert memory["experiences"] == {
        "idxs": [],
        "questions": [],
        "keys": [],
        "trajectories": [],
        "reflections": [],
    }
    assert memory["success_traj_docs"] == []
    assert not memory["vectorstore"]
    assert memory["insights"] == []


def test_gather_experience(hotpotqa_distractor_sample_path: str) -> None:
    """Test gather_experience."""
    hotpotqa = joblib.load(hotpotqa_distractor_sample_path)

    gt_new_experiences = {
        "idxs": [0],
        "questions": [
            "What giant silverware company was started as a religious Utopian group and was for many years run by Pierrepont Noyes?"
        ],
        "keys": ["Oneida Limited"],
        "trajectories": [
            [
                {
                    "react_output": [
                        {
                            "thought": "I need to search for the giant silverware company that was started as a religious Utopian group and was run by Pierrepont Noyes.",
                            "action_type": "Search",
                            "query": "giant silverware company started as religious Utopian group Pierrepont Noyes",
                            "observation": "Search result",
                            "answer": "",
                            "external_tool_info": {
                                "search_result": "Search result",
                                "lookup_result": "",
                            },
                            "is_correct": False,
                        },
                        {
                            "thought": "The search query was too specific. I should try searching for the silverware company and then look for information about its history and founder.",
                            "action_type": "Search",
                            "query": "silverware company Pierrepont Noyes",
                            "observation": "Search result",
                            "answer": "",
                            "external_tool_info": {
                                "search_result": "Search result",
                                "lookup_result": "",
                            },
                            "is_correct": False,
                        },
                        {
                            "thought": "Pierrepont Noyes was the head of Oneida Limited, a silverware company. I need to confirm if Oneida Limited was indeed started as a religious Utopian group.",
                            "action_type": "Search",
                            "query": "Oneida Limited religious Utopian group",
                            "observation": "Search result",
                            "answer": "",
                            "external_tool_info": {
                                "search_result": "Search result",
                                "lookup_result": "",
                            },
                            "is_correct": False,
                        },
                        {
                            "thought": "Oneida Limited was indeed started as a religious Utopian group. The answer to the question is Oneida Limited.",
                            "action_type": "Finish",
                            "query": "Oneida Limited",
                            "observation": "Answer is CORRECT",
                            "answer": "Oneida Limited",
                            "external_tool_info": {
                                "search_result": "",
                                "lookup_result": "",
                            },
                            "is_correct": True,
                        },
                    ],
                    "reflections": [],
                }
            ]
        ],
        "reflections": [[]],
    }

    action_responses = [
        "I need to search for the giant silverware company that was started as a religious Utopian group and was run by Pierrepont Noyes.\nAction: Search[Pierrepont Noyes]\nObservation: Could not find [Pierrepont Noyes]. Similar: ['Noyes Academy', 'Penn Noyes', 'Pierrepont Noyes', 'Pierrepont Noyes House', 'Pierrepont Noyes Mausoleum', 'Pierrepont Noyes Memorial Foundation']\nThought: I need to search for the company directly instead.\nAction: Search[giant silverware company religious Utopian group]\nObservation: Could not find [giant silverware company religious Utopian group]. Similar: ['Oneida Limited', 'Oneida Community', 'Oneida Silver']\nThought: I should search for Oneida Limited.\nAction: Search[Oneida Limited]\nObservation: Oneida Limited is one of the world's largest designers and sellers of stainless steel and silver-plated cutlery and tableware for the consumer and foodservice industries.\nThought: Oneida Limited is the giant silverware company that was started as a religious Utopian group and run by Pierrepont Noyes.\nAction: Finish[Oneida Limited]",
        "Search[giant silverware company started as religious Utopian group Pierrepont Noyes]\nObservation: Could not find [giant silverware company started as religious Utopian group Pierrepont Noyes]. Similar: ['Oneida Community', 'Gorham Silver Manufacturing Company', 'F. B. Rogers Silver Company']\nThought: Let's search for Oneida Community and see if it was started as a religious Utopian group and run by Pierrepont Noyes.\nAction: Search[Oneida Community]\nObservation: The Oneida Community was a religious perfectionist group founded by John Humphrey Noyes in 1848 in Oneida, New York.\nThought: The Oneida Community was indeed a religious Utopian group, but it was founded by John Humphrey Noyes, not Pierrepont Noyes. Let's look up Pierrepont Noyes to see if he was involved with the Oneida Community.\nAction: Lookup[Pierrepont Noyes]\nObservation: (Result 1 / 1) Pierrepont Noyes was an American industrialist and president of Oneida Community from 1877 to 1881.\nThought: Pierrepont Noyes was indeed involved with the Oneida Community. So the answer to the question is Oneida Community.\nAction: Finish[Oneida Community]",
        "The search query was too specific. I should try searching for the silverware company and then look for information about its history and founder.\nAction: Search[silverware company Pierrepont Noyes]\nObservation 2: Oneida Limited is an American manufacturer of tableware and cutlery. The company originated in a utopian community founded in the 19th century.\nThought: Oneida Limited was the silverware company started as a religious Utopian group. Now I need to find out if it was run by Pierrepont Noyes.\nAction: Lookup[Pierrepont Noyes]\nObservation 3: (Result 1 / 1) Pierrepont Noyes was a member of the Oneida Community, a religious communal society founded by his father John Humphrey Noyes.\nThought: Oneida Limited was indeed run by Pierrepont Noyes, so the answer is Oneida Limited.\nAction: Finish[Oneida Limited]",
        "Search[silverware company Pierrepont Noyes]\nObservation 2: Could not find [silverware company Pierrepont Noyes]. Similar: []\nThought: I should try searching for Pierrepont Noyes to see if I can find any information about the company he was associated with.\nAction: Search[Pierrepont Noyes]\nObservation 3: Pierrepont Noyes (1868–1933) was an American industrialist, socialist, and founder of the Oneida Community.\nThought: Pierrepont Noyes was the founder of the Oneida Community. I should search for the history of the Oneida Community to see if it is the silverware company in question.\nAction: Search[Oneida Community silverware company]\nObservation 4: The Oneida Community was a religiously-centered communal society founded by John Humphrey Noyes in 1848 in Oneida, New York. The Oneida Community practiced communalism, complex marriage, male continence, mutual criticism, and asceticism.\nThought: The Oneida Community is the religious Utopian group mentioned in the question. I should now search for the connection between the Oneida Community and the silverware company.\nAction: Search[Oneida Community silverware company connection]\nObservation 5: Could not find [Oneida Community silverware company connection]. Similar: []\nThought: Since I couldn't find a direct connection, I should try searching for the history of the Oneida Community to see if it eventually led to the creation of a silverware company.\nAction: Search[Oneida Community history]\nObservation 6: The Oneida Community dissolved in 1881, and some of its members formed the Oneida Community Limited to continue the manufacture of silverware.\nThought: The Oneida Community eventually led to the formation of the Oneida Community Limited, a silverware company. The answer to the question is Oneida Community Limited.\nAction: Finish[Oneida Community Limited]",
        "Pierrepont Noyes was the head of Oneida Limited, a silverware company. I need to confirm if Oneida Limited was indeed started as a religious Utopian group.\nAction: Search[Oneida Limited history]\nObservation 3: Oneida Limited was founded by John Humphfrey Noyes in the mid-19th century as a religious commune known as the Oneida Community.\nThought: Oneida Limited was indeed started as a religious Utopian group. The answer to the question is Oneida Limited.\nAction: Finish[Oneida Limited]",
        "Search[Oneida Limited religious Utopian group]\nObservation 3: Could not find [Oneida Limited religious Utopian group]. Similar: ['Oneida, New York', 'Oneida Community', 'Oneida Silverware']\nThought: I can search for the history of Oneida Limited to confirm if it was started as a religious Utopian group.\nAction: Search[Oneida Limited history]\nObservation 4: Oneida Limited is one of the world's largest designers and sellers of stainless steel and silverplated cutlery and tableware for the consumer and foodservice industries.\nThought: The search results do not provide information about the religious Utopian origins of Oneida Limited. I should look up the history of the Oneida Community to find the connection.\nAction: Lookup[Oneida Community]\nObservation 5: (Result 1 / 1) The Oneida Community was a religious communal society founded by John Humphrey Noyes in 1848 in Oneida, New York.\nThought: The Oneida Community was a religious communal society founded by John Humphrey Noyes in 1848. Pierrepont Noyes, who later ran Oneida Limited, was brought up in the Oneida Community. Therefore, the giant silverware company started as a religious Utopian group and run by Pierrepont Noyes is Oneida Limited.\nAction: Finish[Oneida Limited]",
        "Oneida Limited was indeed started as a religious Utopian group. The answer to the question is Oneida Limited.\nAction: Finish[Oneida Limited]",
        "Finish[Oneida Limited]",
    ]
    llm = FakeListChatModel(responses=action_responses)
    agent = ExpeLAgent(llm=llm, mode={"qa": "hotpotqa"})
    agent.reflexion_react_agent.strategy.docstore.search = lambda x: "Search result"
    agent.reflexion_react_agent.strategy.docstore.lookup = lambda x: "Lookup result"
    new_experiences = agent.gather_experience(
        questions=hotpotqa.question.values[-1:], keys=hotpotqa.answer.values[-1:]
    )

    assert new_experiences == gt_new_experiences
    assert new_experiences == agent.experience_memory.experiences
    assert len(agent.experience_memory.success_traj_docs) == 13
    assert agent.experience_memory.vectorstore


def test_update_insights() -> None:
    """Test update_insights."""
    insights = [
        {"insight": "Test 1", "score": 1},
        {"insight": "Test 2", "score": 2},
        {"insight": "Test 3", "score": 3},
    ]
    memory = ExpeLInsightMemory(insights, max_num_insights=3)
    llm = FakeListChatModel(responses=[])
    agent = ExpeLAgent(llm=llm, mode={"qa": "hotpotqa"}, insight_memory=memory)

    # Valid remove.
    gt_insights = [{"insight": "Test 2", "score": 2}, {"insight": "Test 3", "score": 3}]
    agent.update_insights(
        [
            ("REMOVE 0", "Test 1"),
        ]
    )
    assert agent.insight_memory.insights == gt_insights

    # Invalid remove.
    agent.update_insights(
        [
            ("REMOVE 0", "Test askdasf"),
        ]
    )
    assert agent.insight_memory.insights == gt_insights

    # Valid agree.
    gt_insights = [{"insight": "Test 2", "score": 3}, {"insight": "Test 3", "score": 3}]
    agent.update_insights([("AGREE 0", "Test 2")])
    assert agent.insight_memory.insights == gt_insights

    # Invalid agree.
    agent.update_insights([("AGREE 0", "Test asdjafh")])
    assert agent.insight_memory.insights == gt_insights

    # Edit.
    gt_insights = [{"insight": "Test 2", "score": 3}, {"insight": "Test 4", "score": 4}]
    agent.update_insights([("EDIT 1", "Test 4")])
    assert agent.insight_memory.insights == gt_insights

    # Add.
    gt_insights = [
        {"insight": "Test 2", "score": 3},
        {"insight": "Test 4", "score": 4},
        {"insight": "Another insight", "score": 2},
    ]
    agent.update_insights([("ADD", "Another insight")])
    assert agent.insight_memory.insights == gt_insights


def test_extract_insights(expel_experiences_10_fake_path: str) -> None:
    """Test extract_insights."""
    experiences = joblib.load(expel_experiences_10_fake_path)
    selected_indices = [3]
    selected_dict = {
        key: [value[i] for i in selected_indices] for key, value in experiences.items()
    }
    selected_dict["idxs"] = list(range(len(selected_indices)))

    gt_insights = [
        {
            "insight": "Always try multiple variations of search terms when looking for specific information.",
            "score": 2,
        },
        {
            "insight": "If unable to find relevant information through initial searches, consider looking for official announcements or press releases from the company.",
            "score": 2,
        },
    ]
    responses = [
        "ADD 11: Always try multiple variations of search terms when looking for specific information.\nADD 12: If unable to find relevant information through initial searches, consider looking for official announcements or press releases from the company.\nREMOVE 3: Always use the exact search term provided in the question, do not try variations.\nEDIT 7: Make sure to exhaust all possible search options before concluding that the information is unavailable.",
    ]
    llm = FakeListChatModel(responses=responses)
    agent = ExpeLAgent(llm=llm, mode={"qa": "hotpotqa"})
    agent.extract_insights(selected_dict)
    assert agent.insight_memory.insights == gt_insights


def test_generate(expel_experiences_10_fake_path: str) -> None:
    """Test generate."""
    experiences = joblib.load(expel_experiences_10_fake_path)
    question = "What giant silverware company was started as a religious Utopian group and was for many years run by Pierrepont Noyes?"
    key = "Oneida Limited"

    gt_out = {
        "idxs": [0],
        "questions": [
            "What giant silverware company was started as a religious Utopian group and was for many years run by Pierrepont Noyes?"
        ],
        "keys": ["Oneida Limited"],
        "trajectories": [
            [
                {
                    "react_output": [
                        {
                            "thought": "I need to find out the name of the giant silverware company that was started as a religious Utopian group and was run by Pierrepont Noyes for many years. I should search for the company's name and its history to gather more information.",
                            "action_type": "Search",
                            "query": "giant silverware company religious Utopian group Pierrepont Noyes",
                            "observation": "Search result",
                            "answer": "",
                            "external_tool_info": {
                                "search_result": "Search result",
                                "lookup_result": "",
                            },
                            "is_correct": False,
                        },
                        {
                            "thought": "I need to search for more information about Oneida Limited to see if it matches the description of a giant silverware company that was started as a religious Utopian group and was run by Pierrepont Noyes. I should look for details about the company's history and any relevant connections to Noyes.",
                            "action_type": "Search",
                            "query": "Oneida Limited giant silverware company religious Utopian group Pierrepont Noyes",
                            "observation": "Search result",
                            "answer": "",
                            "external_tool_info": {
                                "search_result": "Search result",
                                "lookup_result": "",
                            },
                            "is_correct": False,
                        },
                        {
                            "thought": 'It seems like the search results are not providing specific information about Oneida Limited as a giant silverware company. I should try to look up the keyword "silverware" in the last passage successfully found and see if it provides any relevant details about Oneida Limited\'s products.',
                            "action_type": "Lookup",
                            "query": "silverware",
                            "observation": "Lookup result",
                            "answer": "",
                            "external_tool_info": {
                                "search_result": "",
                                "lookup_result": "Lookup result",
                            },
                            "is_correct": False,
                        },
                        {
                            "thought": "The passage mentions that Oneida Limited specialized in silverware and stainless steel cutlery under Pierrepont Noyes' leadership. This aligns with the description of a giant silverware company. I can confidently conclude that Oneida Limited is the giant silverware company started as a religious Utopian group and run by Pierrepont Noyes for many years. I need to finish with this answer.",
                            "action_type": "Finish",
                            "query": "Oneida Limited",
                            "observation": "Answer is CORRECT",
                            "answer": "Oneida Limited",
                            "external_tool_info": {
                                "search_result": "",
                                "lookup_result": "",
                            },
                            "is_correct": True,
                        },
                    ],
                    "reflections": [],
                }
            ]
        ],
        "reflections": [[]],
    }

    gt_insights = [
        {
            "insight": "When searching for information, if the initial search query does not return relevant results, try using different keywords or search terms to refine the search.",
            "score": 2,
        },
        {
            "insight": "When searching for information, consider searching for the creator or key figures related to the topic to gather more relevant details.",
            "score": 2,
        },
        {
            "insight": "When searching for information, consider looking up keywords or terms mentioned in the search results to gather more specific details.",
            "score": 2,
        },
    ]

    responses = [
        "ADD 9: When searching for information, if the initial search query does not return relevant results, try using different keywords or search terms to refine the search.",
        "ADD 1: When searching for information, consider searching for the creator or key figures related to the topic to gather more relevant details.",
        "I need to find out the name of the giant silverware company that was started as a religious Utopian group and was run by Pierrepont Noyes for many years. I should search for the company's name and its history to gather more information. \nAction: Search[giant silverware company religious Utopian Pierrepont Noyes]\nObservation: The search did not return relevant results. I should try refining my search terms to find the information I need.\nAction: Search[silverware company Pierrepont Noyes history]\nObservation: The search results mention Oneida Limited as the giant silverware company that was started as a religious Utopian group and was run by Pierrepont Noyes for many years. \nAction: Lookup[Oneida Limited]\nObservation: Oneida Limited was a company that originated in the 19th century as a religious Utopian community, the Oneida Community, founded by John Humphrey Noyes in Oneida, New York. The company initially produced animal traps and then shifted to the production of silverware. It became known for its high-quality silverware and cutlery products. \nAction: Finish[Oneida Limited]\nObservation: Answer is CORRECT.",
        "Search[giant silverware company religious Utopian group Pierrepont Noyes]",
        "I need to search for more information about Oneida Limited to see if it matches the description of a giant silverware company that was started as a religious Utopian group and was run by Pierrepont Noyes. I should look for details about the company's history and any relevant connections to Noyes. \nAction: Search[Oneida Limited history Pierrepont Noyes]\nObservation 2: Oneida Limited is an American manufacturer of tableware and cutlery. The company originated in a utopian community, the Oneida Community, established in the 19th century. Pierrepont Burt Noyes, a descendant of the community's founder, became the head of Oneida Limited and led the company for many years.\nThought: It seems like Oneida Limited fits the description of the giant silverware company started as a religious Utopian group and run by Pierrepont Noyes. I should look for more specific details to confirm this information.\nAction: Lookup[silverware]\nObservation 3: Oneida Limited is known for its production of silverware, flatware, and tableware products. The company has a long history of manufacturing high-quality silverware and cutlery items.\nThought: Based on the information gathered, Oneida Limited is indeed the giant silverware company that matches the description provided in the question. I can now finish with this answer.\nAction: Finish[Oneida Limited] \nObservation: Answer is CORRECT",
        "Search[Oneida Limited giant silverware company religious Utopian group Pierrepont Noyes]",
        'It seems like the search results are not providing specific information about Oneida Limited as a giant silverware company. I should try to look up the keyword "silverware" in the last passage successfully found and see if it provides any relevant details about Oneida Limited\'s products.\nAction: Lookup[silverware]',
        "Lookup[silverware]",
        "The passage mentions that Oneida Limited specialized in silverware and stainless steel cutlery under Pierrepont Noyes' leadership. This aligns with the description of a giant silverware company. I can confidently conclude that Oneida Limited is the giant silverware company started as a religious Utopian group and run by Pierrepont Noyes for many years. I need to finish with this answer.\nAction: Finish[Oneida Limited]\nObservation: Answer is CORRECT",
        "Finish[Oneida Limited]",
        "ADD 2: When searching for information, consider looking up keywords or terms mentioned in the search results to gather more specific details.",
    ]

    agent = ExpeLAgent(
        llm=FakeListChatModel(responses=responses),
        mode={"qa": "hotpotqa"},
        experience_memory=ExpeLExperienceMemory(experiences),
    )
    agent.reflexion_react_agent.strategy.docstore.search = lambda x: "Search result"
    agent.reflexion_react_agent.strategy.docstore.lookup = lambda x: "Lookup result"
    out = agent.generate(question=question, key=key)
    assert out == gt_out
    assert len(agent.experience_memory.experiences["idxs"]) == 6
    assert agent.experience_memory.experiences["questions"][5] == question
    assert agent.experience_memory.experiences["keys"][5] == key
    assert agent.experience_memory.experiences["reflections"][5] == []
    assert agent.insight_memory.insights == gt_insights
    assert len(agent.experience_memory.success_traj_docs) == 23
    assert agent.experience_memory.vectorstore
