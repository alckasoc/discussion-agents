"""Unit tests for ExpeL."""
import joblib

from langchain_community.chat_models.fake import FakeListChatModel
from langchain_core.language_models.chat_models import BaseChatModel
from discussion_agents.cog.agent.reflexion import ReflexionReActAgent

from discussion_agents.cog.agent.expel import ExpeLAgent
from discussion_agents.cog.modules.memory.expel import ExpeLExperienceMemory, ExpeLInsightMemory

def test_init(expel_experiences_10_fake_path: str) -> None:
    """Test initialization."""
    llm = FakeListChatModel(responses=["1"])

    agent = ExpeLAgent(
        llm=llm,
    )
    assert isinstance(agent.llm, BaseChatModel)
    assert isinstance(agent.reflexion_react_agent, ReflexionReActAgent)
    assert isinstance(agent.experience_memory, ExpeLExperienceMemory)
    assert isinstance(agent.insight_memory, ExpeLInsightMemory)
    assert agent.success_batch_size == 8
    assert agent.experience_memory.experiences == {'idxs': [], 'questions': [], 'keys': [], 'trajectories': [], 'reflections': []}
    assert not agent.experience_memory.success_traj_docs
    assert not agent.experience_memory.vectorstore
    assert not agent.insight_memory.insights

    # Test with all parameters specified except experience memory and reflexion_react_agent.
    agent = ExpeLAgent(
        llm=llm,
        self_reflect_llm=FakeListChatModel(responses=["2"]),
        action_llm=FakeListChatModel(responses=["3"]),
        reflexion_react_kwargs={"max_steps": 3},
        insight_memory=ExpeLInsightMemory(insights=[{"insight": "blah blah", "score": 10}]),
        success_batch_size=10
    )
    assert isinstance(agent.llm, BaseChatModel)
    assert isinstance(agent.reflexion_react_agent, ReflexionReActAgent)
    assert isinstance(agent.experience_memory, ExpeLExperienceMemory)
    assert isinstance(agent.insight_memory, ExpeLInsightMemory)
    assert agent.success_batch_size == 10
    assert agent.reflexion_react_agent.self_reflect_llm.responses == ["2"]
    assert agent.reflexion_react_agent.action_llm.responses == ["3"]
    assert agent.reflexion_react_agent.max_steps == 3
    assert agent.experience_memory.experiences == {'idxs': [], 'questions': [], 'keys': [], 'trajectories': [], 'reflections': []}
    assert not agent.experience_memory.success_traj_docs
    assert not agent.experience_memory.vectorstore
    assert agent.insight_memory.insights == [{"insight": "blah blah", "score": 10}]

    # Test with custom reflexion_react_agent (verify it overrides reflexion_react_kwargs)
    agent = ExpeLAgent(
        llm=llm,
        reflexion_react_kwargs={"max_steps": 100},
        reflexion_react_agent=ReflexionReActAgent(self_reflect_llm=llm, action_llm=llm),
    )
    assert agent.reflexion_react_agent.max_steps == 6

    # Test with custom experience memory (verify correct initialization).
    experiences = joblib.load(expel_experiences_10_fake_path)
    experiences = {key: value[:1] for key, value in experiences.items()}

    agent = ExpeLAgent(
        llm=llm,
        experience_memory=ExpeLExperienceMemory(experiences)
    )
    assert agent.experience_memory.experiences == experiences
    assert agent.insight_memory.insights == []


def test_reset(expel_agent: ExpeLAgent) -> None:
    """Test reset."""
    expel_agent.reflexion_react_agent.memory.scratchpad == "cat"
    expel_agent.experience_memory.experiences == "dog"
    expel_agent.insight_memory.insights = ["turtle"]
    expel_agent.reset()
    assert expel_agent.reflexion_react_agent.memory.scratchpad == ""
    assert expel_agent.experience_memory.experiences == {'idxs': [], 'questions': [], 'keys': [], 'trajectories': [], 'reflections': []}
    assert expel_agent.insight_memory.insights == []


def test_retrieve(expel_agent: ExpeLAgent) -> None:
    """Test retrieve."""
    memory = expel_agent.retrieve()
    assert list(memory.keys()) == ["experiences", "success_traj_docs", "vectorstore", "insights"]
    assert memory['experiences'] == {'idxs': [], 'questions': [], 'keys': [], 'trajectories': [], 'reflections': []}
    assert memory['success_traj_docs'] == []
    assert not memory['vectorstore']
    assert memory['insights'] == []


def test_gather_experience(hotpotqa_distractor_sample_path: str) -> None:
    """Test gather_experience."""
    hotpotqa = joblib.load(hotpotqa_distractor_sample_path)

    gt_new_experiences = {
        'idxs': [0],
        'questions': ['What giant silverware company was started as a religious Utopian group and was for many years run by Pierrepont Noyes?'],
        'keys': ['Oneida Limited'],
        'trajectories': [[(True,
            'Oneida Limited',
            [('Thought: I need to search for the giant silverware company that was started as a religious Utopian group and was run by Pierrepont Noyes.',
            'Action: Search[giant silverware company started as religious Utopian group Pierrepont Noyes]',
            'Observation 1: Could not find [giant silverware company started as religious Utopian group Pierrepont Noyes]. Similar: []'),
            ('Thought: The search query was too specific. I should try searching for the silverware company and then look for information about its history and founder.',
            'Action: Search[silverware company Pierrepont Noyes]',
            'Observation 2: Pierrepont Burt Noyes (August 18, 1870 – April 15, 1959) was an American businessman and writer. He was brought up in the Oneida Community, a religious Utopian group. Noyes later became the head of Oneida Limited, a position he held for many years.'),
            ('Thought: Pierrepont Noyes was the head of Oneida Limited, a silverware company. I need to confirm if Oneida Limited was indeed started as a religious Utopian group.',
            'Action: Search[Oneida Limited religious Utopian group]',
            "Observation 3: The Oneida Community ( oh-NYE-də) was a perfectionist religious communal society founded by John Humphrey Noyes and his followers in 1848 near Oneida, New York. The community believed that Jesus had already returned in AD 70, making it possible for them to bring about Jesus's millennial kingdom themselves, and be perfect and free of sin in this world, not just in Heaven (a belief called perfectionism). The Oneida Community practiced communalism (in the sense of communal property and possessions), group marriage, male sexual continence, Oneida stirpiculture (a form of eugenics), and mutual criticism.The community's original 87 members grew to 172 by February 1850, 208 by 1852, and 306 by 1878. There were smaller Noyesian communities in Wallingford, Connecticut; Newark, New Jersey; Putney and Cambridge, Vermont. The branches were closed in 1854 except for the Wallingford branch, which operated until the 1878 tornado devastated it.The Oneida Community dissolved in 1881, converting itself to a joint-stock company. This eventually became the silverware company Oneida Limited, one of the largest in the world."),
            ('Thought: Oneida Limited was indeed started as a religious Utopian group. The answer to the question is Oneida Limited.',
            'Action: Finish[Oneida Limited]',
            'Observation 4: Answer is CORRECT')])]],
        'reflections': [[]]
    }
    self_reflect_responses = ["1"]
    action_responses = [
        "I need to search for the giant silverware company that was started as a religious Utopian group and was run by Pierrepont Noyes.\nAction: Search[Pierrepont Noyes]\nObservation: Could not find [Pierrepont Noyes]. Similar: ['Noyes Academy', 'Penn Noyes', 'Pierrepont Noyes', 'Pierrepont Noyes House', 'Pierrepont Noyes Mausoleum', 'Pierrepont Noyes Memorial Foundation']\nThought: I need to search for the company directly instead.\nAction: Search[giant silverware company religious Utopian group]\nObservation: Could not find [giant silverware company religious Utopian group]. Similar: ['Oneida Limited', 'Oneida Community', 'Oneida Silver']\nThought: I should search for Oneida Limited.\nAction: Search[Oneida Limited]\nObservation: Oneida Limited is one of the world's largest designers and sellers of stainless steel and silver-plated cutlery and tableware for the consumer and foodservice industries.\nThought: Oneida Limited is the giant silverware company that was started as a religious Utopian group and run by Pierrepont Noyes.\nAction: Finish[Oneida Limited]",
        "Search[giant silverware company started as religious Utopian group Pierrepont Noyes]\nObservation: Could not find [giant silverware company started as religious Utopian group Pierrepont Noyes]. Similar: ['Oneida Community', 'Gorham Silver Manufacturing Company', 'F. B. Rogers Silver Company']\nThought: Let's search for Oneida Community and see if it was started as a religious Utopian group and run by Pierrepont Noyes.\nAction: Search[Oneida Community]\nObservation: The Oneida Community was a religious perfectionist group founded by John Humphrey Noyes in 1848 in Oneida, New York.\nThought: The Oneida Community was indeed a religious Utopian group, but it was founded by John Humphrey Noyes, not Pierrepont Noyes. Let's look up Pierrepont Noyes to see if he was involved with the Oneida Community.\nAction: Lookup[Pierrepont Noyes]\nObservation: (Result 1 / 1) Pierrepont Noyes was an American industrialist and president of Oneida Community from 1877 to 1881.\nThought: Pierrepont Noyes was indeed involved with the Oneida Community. So the answer to the question is Oneida Community.\nAction: Finish[Oneida Community]",
        'The search query was too specific. I should try searching for the silverware company and then look for information about its history and founder.\nAction: Search[silverware company Pierrepont Noyes]\nObservation 2: Oneida Limited is an American manufacturer of tableware and cutlery. The company originated in a utopian community founded in the 19th century.\nThought: Oneida Limited was the silverware company started as a religious Utopian group. Now I need to find out if it was run by Pierrepont Noyes.\nAction: Lookup[Pierrepont Noyes]\nObservation 3: (Result 1 / 1) Pierrepont Noyes was a member of the Oneida Community, a religious communal society founded by his father John Humphrey Noyes.\nThought: Oneida Limited was indeed run by Pierrepont Noyes, so the answer is Oneida Limited.\nAction: Finish[Oneida Limited]',
        "Search[silverware company Pierrepont Noyes]\nObservation 2: Could not find [silverware company Pierrepont Noyes]. Similar: []\nThought: I should try searching for Pierrepont Noyes to see if I can find any information about the company he was associated with.\nAction: Search[Pierrepont Noyes]\nObservation 3: Pierrepont Noyes (1868–1933) was an American industrialist, socialist, and founder of the Oneida Community.\nThought: Pierrepont Noyes was the founder of the Oneida Community. I should search for the history of the Oneida Community to see if it is the silverware company in question.\nAction: Search[Oneida Community silverware company]\nObservation 4: The Oneida Community was a religiously-centered communal society founded by John Humphrey Noyes in 1848 in Oneida, New York. The Oneida Community practiced communalism, complex marriage, male continence, mutual criticism, and asceticism.\nThought: The Oneida Community is the religious Utopian group mentioned in the question. I should now search for the connection between the Oneida Community and the silverware company.\nAction: Search[Oneida Community silverware company connection]\nObservation 5: Could not find [Oneida Community silverware company connection]. Similar: []\nThought: Since I couldn't find a direct connection, I should try searching for the history of the Oneida Community to see if it eventually led to the creation of a silverware company.\nAction: Search[Oneida Community history]\nObservation 6: The Oneida Community dissolved in 1881, and some of its members formed the Oneida Community Limited to continue the manufacture of silverware.\nThought: The Oneida Community eventually led to the formation of the Oneida Community Limited, a silverware company. The answer to the question is Oneida Community Limited.\nAction: Finish[Oneida Community Limited]",
        'Pierrepont Noyes was the head of Oneida Limited, a silverware company. I need to confirm if Oneida Limited was indeed started as a religious Utopian group.\nAction: Search[Oneida Limited history]\nObservation 3: Oneida Limited was founded by John Humphfrey Noyes in the mid-19th century as a religious commune known as the Oneida Community.\nThought: Oneida Limited was indeed started as a religious Utopian group. The answer to the question is Oneida Limited.\nAction: Finish[Oneida Limited]',
        "Search[Oneida Limited religious Utopian group]\nObservation 3: Could not find [Oneida Limited religious Utopian group]. Similar: ['Oneida, New York', 'Oneida Community', 'Oneida Silverware']\nThought: I can search for the history of Oneida Limited to confirm if it was started as a religious Utopian group.\nAction: Search[Oneida Limited history]\nObservation 4: Oneida Limited is one of the world's largest designers and sellers of stainless steel and silverplated cutlery and tableware for the consumer and foodservice industries.\nThought: The search results do not provide information about the religious Utopian origins of Oneida Limited. I should look up the history of the Oneida Community to find the connection.\nAction: Lookup[Oneida Community]\nObservation 5: (Result 1 / 1) The Oneida Community was a religious communal society founded by John Humphrey Noyes in 1848 in Oneida, New York.\nThought: The Oneida Community was a religious communal society founded by John Humphrey Noyes in 1848. Pierrepont Noyes, who later ran Oneida Limited, was brought up in the Oneida Community. Therefore, the giant silverware company started as a religious Utopian group and run by Pierrepont Noyes is Oneida Limited.\nAction: Finish[Oneida Limited]",
        'Oneida Limited was indeed started as a religious Utopian group. The answer to the question is Oneida Limited.\nAction: Finish[Oneida Limited]',
        'Finish[Oneida Limited]'
    ]
    self_reflect_llm = FakeListChatModel(responses=self_reflect_responses)
    action_llm = FakeListChatModel(responses=action_responses)
    agent = ExpeLAgent(llm=action_llm, self_reflect_llm=self_reflect_llm, action_llm=action_llm)
    new_experiences = agent.gather_experience(
        questions=hotpotqa.question.values[-1:],
        keys=hotpotqa.answer.values[-1:]
    )
    assert new_experiences == gt_new_experiences
    assert new_experiences == agent.experience_memory.experiences
    assert len(agent.experience_memory.success_traj_docs) == 13
    assert agent.experience_memory.vectorstore

def test_update_insights() -> None:
    """"Test update_insights."""
    insights = [{"insight": "Test 1", "score": 1}, {"insight": "Test 2", "score": 2}, {"insight": "Test 3", "score": 3}]
    memory = ExpeLInsightMemory(insights, max_num_insights=3)
    llm = FakeListChatModel(responses=['1'])
    agent = ExpeLAgent(llm=llm, insight_memory=memory)

    # Valid remove.
    gt_insights = [{'insight': 'Test 2', 'score': 2}, {'insight': 'Test 3', 'score': 3}]
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
    gt_insights = [{'insight': 'Test 2', 'score': 3}, {'insight': 'Test 3', 'score': 3}]
    agent.update_insights(
        [
            ("AGREE 0", "Test 2")
        ]
    )
    assert agent.insight_memory.insights == gt_insights

    # Invalid agree.
    agent.update_insights(
        [
            ("AGREE 0", "Test asdjafh")
        ]
    )
    assert agent.insight_memory.insights == gt_insights
    
    # Edit.
    gt_insights = [{'insight': 'Test 4', 'score': 4}, {'insight': 'Test 3', 'score': 3}]
    agent.update_insights(
        [
            ("EDIT 1", "Test 4")
        ]
    )
    assert agent.insight_memory.insights == gt_insights

    # Add.
    gt_insights = [
        {'insight': 'Test 4', 'score': 4},
        {'insight': 'Test 3', 'score': 3},
        {'insight': 'Another insight', 'score': 2}
    ]
    agent.update_insights(
        [
            ("ADD", "Another insight")
        ]
    )
    assert agent.insight_memory.insights == gt_insights


def test_extract_insights(expel_experiences_10_fake_path: str) -> None:
    """Test extract_insights."""
    experiences = joblib.load(expel_experiences_10_fake_path)
    selected_indices = [6, 3, 0]
    selected_dict = {key: [value[i] for i in selected_indices] for key, value in experiences.items()}
    selected_dict['idxs'] = list(range(len(selected_indices)))
    
    gt_insights = [
        {'insight': 'Always try multiple variations of search terms when looking for specific information.',
        'score': 2},
        {'insight': 'If unable to find relevant information through initial searches, consider looking for official announcements or press releases from the company.',
        'score': 2},
        {'insight': 'Consider reaching out directly to the company or checking their official website for specific information if initial searches do not yield relevant results.',
        'score': 2},
        {'insight': 'When searching for specific information, consider looking for biographical information about relevant individuals to gather additional context and insights.',
        'score': 2}
    ]
    responses = [
        'ADD 11: Always try multiple variations of search terms when looking for specific information.\nADD 12: If unable to find relevant information through initial searches, consider looking for official announcements or press releases from the company.\nREMOVE 3: Always use the exact search term provided in the question, do not try variations.\nEDIT 7: Make sure to exhaust all possible search options before concluding that the information is unavailable.',
        'ADD 3: Consider reaching out directly to the company or checking their official website for specific information if initial searches do not yield relevant results.',
        'ADD 4: When searching for specific information, consider looking for biographical information about relevant individuals to gather additional context and insights.'
    ]
    llm = FakeListChatModel(responses=responses)
    agent = ExpeLAgent(llm=llm)
    agent.extract_insights(selected_dict)
    assert agent.insight_memory.insights == gt_insights

def test_generate(expel_experiences_10_fake_path: str) -> None:
    """Test generate."""
    experiences = joblib.load(expel_experiences_10_fake_path)
    question = 'What giant silverware company was started as a religious Utopian group and was for many years run by Pierrepont Noyes?'
    key = 'Oneida Limited'
    
    gt_out = {
        'idxs': [0],
        'questions': ['What giant silverware company was started as a religious Utopian group and was for many years run by Pierrepont Noyes?'],
        'keys': ['Oneida Limited'],
        'trajectories': [[(True,
            'Oneida Limited',
            [('Thought: I need to search for the giant silverware company that was started as a religious Utopian group and was run by Pierrepont Noyes.',
            'Action: Search[giant silverware company religious Utopian Pierrepont Noyes]',
            'Observation 1: Pierrepont Burt Noyes (August 18, 1870 – April 15, 1959) was an American businessman and writer. He was brought up in the Oneida Community, a religious Utopian group. Noyes later became the head of Oneida Limited, a position he held for many years.'),
            ('Thought: The giant silverware company started as a religious Utopian group is Oneida Limited, run by Pierrepont Noyes. Now, I need to find out more information about Oneida Limited. ',
            'Action: Search[Oneida Limited]',
            "Observation 2: Oneida Limited () is an American manufacturer and seller of tableware and cutlery. Oneida is one of the world's largest designers and sellers of stainless steel and silverplated cutlery and tableware for the consumer and foodservice industries. It is also the largest supplier of dinnerware to the foodservice industry in North America. The company operates in the United States, Canada, Latin America, Europe, and Asia, marketing and distributing tabletop products, which include flatware, dinnerware, crystal stemware, glassware and kitchen tools and gadgets.  The factory in upstate NY was sold to Liberty Tabletop, who is the sole manufacturer of US made flatware. The company originated in the late-nineteenth century in the Oneida Community in Oneida, New York."),
            ('Thought: Oneida Limited was indeed the giant silverware company started as a religious Utopian group and run by Pierrepont Noyes. I have found the information needed to answer the question. ',
            'Action: Finish[Oneida Limited]',
            'Observation 3: Answer is CORRECT')])]],
        'reflections': [[]]
    }
    gt_insights = [
        {'insight': 'Ensure to search for specific information directly related to the question asked, avoiding unnecessary details.',
        'score': 5},
        {'insight': 'Ensure to use specific search terms when looking for information, rather than general terms.',
        'score': 2},
        {'insight': 'When searching for specific information, consider using keywords related to the topic to ensure relevant results are obtained.',
        'score': 2},
        {'insight': 'When searching for specific information, use keywords related to the topic to ensure relevant results are obtained.',
        'score': 2},
        {'insight': 'Verify the accuracy of the obtained information before proceeding with the answer.',
        'score': 2},
        {'insight': 'Focus on directly searching for the specific information needed instead of getting sidetracked by related but not directly relevant information.',
        'score': 2},
        {'insight': 'Prioritize using direct sources of information such as official announcements, press releases, and company websites when searching for specific details.',
        'score': 2},
        {'insight': 'When searching for specific information, consider using different search terms and approaches if initial searches do not yield relevant results.',
        'score': 2},
        {'insight': 'Prioritize using direct sources of information such as official announcements, press releases, and company websites when searching for specific details related to the task at hand.',
        'score': 2},
        {'insight': 'Prioritize using specific search terms and keywords related to the topic to ensure accurate and relevant results are obtained.',
        'score': 2},
        {'insight': 'Focus on efficiently summarizing key information obtained from search results to answer the question accurately.',
        'score': 2}
    ]
    gt_trajectory = [(True,
        'Oneida Limited',
        [('Thought: I need to search for the giant silverware company that was started as a religious Utopian group and was run by Pierrepont Noyes.',
            'Action: Search[giant silverware company religious Utopian Pierrepont Noyes]',
            'Observation 1: Pierrepont Burt Noyes (August 18, 1870 – April 15, 1959) was an American businessman and writer. He was brought up in the Oneida Community, a religious Utopian group. Noyes later became the head of Oneida Limited, a position he held for many years.'),
        ('Thought: The giant silverware company started as a religious Utopian group is Oneida Limited, run by Pierrepont Noyes. Now, I need to find out more information about Oneida Limited. ',
            'Action: Search[Oneida Limited]',
            "Observation 2: Oneida Limited () is an American manufacturer and seller of tableware and cutlery. Oneida is one of the world's largest designers and sellers of stainless steel and silverplated cutlery and tableware for the consumer and foodservice industries. It is also the largest supplier of dinnerware to the foodservice industry in North America. The company operates in the United States, Canada, Latin America, Europe, and Asia, marketing and distributing tabletop products, which include flatware, dinnerware, crystal stemware, glassware and kitchen tools and gadgets.  The factory in upstate NY was sold to Liberty Tabletop, who is the sole manufacturer of US made flatware. The company originated in the late-nineteenth century in the Oneida Community in Oneida, New York."),
        ('Thought: Oneida Limited was indeed the giant silverware company started as a religious Utopian group and run by Pierrepont Noyes. I have found the information needed to answer the question. ',
            'Action: Finish[Oneida Limited]',
            'Observation 3: Answer is CORRECT')])
    ]
    llm_responses = [
        # Initial insight extraction.
        'ADD 11: Ensure to directly search for the specific information needed instead of trying to find similar entities.',
        'ADD 2: Ensure to use specific search terms when looking for information, rather than general terms.\nEDIT 1: Ensure to directly search for the specific information needed instead of trying to find similar entities by using specific search terms.\nREMOVE 1: <EXISTING RULE>',
        'ADD 3: Always verify if the initial search term is specific enough to yield relevant results.\nEDIT 1: Ensure to directly search for the specific information needed instead of trying to find similar entities by using specific search terms and relevant keywords.\nREMOVE 2: This rule is similar to rule 1 and can be integrated into it for clarity and simplicity.',
        'ADD 4: When searching for specific information, consider using keywords related to the topic to ensure relevant results are obtained.',
        'ADD 5: Use specific search terms and relevant keywords when looking for information, rather than general terms.\nADD 6: Verify the accuracy of the obtained information before proceeding with the answer.\nREMOVE 3: Always verify if the initial search term is specific enough to yield relevant results.',
        'ADD 7: Focus on directly searching for the specific information needed instead of getting sidetracked by related but not directly relevant information.',
        'REMOVE 3: Always verify if the initial search term is specific enough to yield relevant results.\nEDIT 4: When searching for specific information, use keywords related to the topic to ensure relevant results are obtained.\nADD 8: Prioritize using direct sources of information such as official announcements, press releases, and company websites when searching for specific details.',
        'ADD 8: When searching for specific information, consider using different search terms and approaches if initial searches do not yield relevant results.',
        'ADD 9: Prioritize using direct sources of information such as official announcements, press releases, and company websites when searching for specific details related to the task at hand.',
        'ADD 10: Prioritize using specific search terms and keywords related to the topic to ensure accurate and relevant results are obtained.',
        # Extract insight from new question.
        'ADD 11: Focus on efficiently summarizing key information obtained from search results to answer the question accurately.\nEDIT 1: Ensure to search for specific information directly related to the question asked, avoiding unnecessary details.\nREMOVE 4: When searching for specific information, use keywords related to the topic to ensure relevant results are obtained.' 
    ]
    self_reflect_llm_responses = []
    action_llm_responses = [
        'I need to search for the giant silverware company that was started as a religious Utopian group and was run by Pierrepont Noyes.\nAction: Search[giant silverware company religious Utopian group Pierrepont Noyes]',
        'Search[giant silverware company religious Utopian Pierrepont Noyes]',
        "The giant silverware company started as a religious Utopian group is Oneida Limited, run by Pierrepont Noyes. Now, I need to find out more information about Oneida Limited. \nAction: Search[Oneida Limited]\nObservation 2: Oneida Limited is one of the world's largest designers and sellers of stainless steel and silver-plated cutlery and tableware for the consumer and foodservice industries. The company originated in a utopian community, the Oneida Community, founded by John Humphrey Noyes in the mid-19th century. Pierrepont Burt Noyes, the son of John Humphrey Noyes, later became the head of Oneida Limited, overseeing its growth and success in the silverware industry.\nThought:\nNow that I have found the information about Oneida Limited, I can provide the answer. \nAction: Finish[Oneida Limited]\nObservation 3: Answer is CORRECT.",
        'Search[Oneida Limited]',
        'Oneida Limited was indeed the giant silverware company started as a religious Utopian group and run by Pierrepont Noyes. I have found the information needed to answer the question. \nAction: Finish[Oneida Limited] \nObservation: Answer is CORRECT.',
        'Finish[Oneida Limited]' 
    ]

    agent = ExpeLAgent(
        llm=FakeListChatModel(responses=llm_responses), 
        reflexion_react_agent=ReflexionReActAgent(
            self_reflect_llm=FakeListChatModel(responses=self_reflect_llm_responses),
            action_llm=FakeListChatModel(responses=action_llm_responses)
        ),
        experience_memory=ExpeLExperienceMemory(experiences)
    )

    out = agent.generate(
        question=question,
        key=key
    )
    assert out == gt_out
    assert len(agent.experience_memory.experiences['idxs']) == 11
    assert agent.experience_memory.experiences['questions'][10] == question
    assert agent.experience_memory.experiences['keys'][10] == key
    assert agent.experience_memory.experiences['reflections'][10] == []
    assert agent.experience_memory.experiences['trajectories'][10] == gt_trajectory
    assert agent.insight_memory.insights == gt_insights
    assert len(agent.experience_memory.success_traj_docs) == 48
    assert agent.experience_memory.vectorstore

    gt_experiences = {
        'idxs': [0],
        'questions': ['What giant silverware company was started as a religious Utopian group and was for many years run by Pierrepont Noyes?'],
        'keys': ['Oneida Limited'],
        'trajectories': [[(True,
            'Oneida Limited',
            [('Thought: I need to search for the giant silverware company that was started as a religious Utopian group and was run by Pierrepont Noyes.',
            'Action: Search[Pierrepont Noyes]',
            'Observation 1: Pierrepont Burt Noyes (August 18, 1870 – April 15, 1959) was an American businessman and writer. He was brought up in the Oneida Community, a religious Utopian group. Noyes later became the head of Oneida Limited, a position he held for many years.'),
            ('Thought: Pierrepont Noyes was the head of Oneida Limited, which was started as a religious Utopian group. I need to search for more information about Oneida Limited to confirm if it is the giant silverware company.',
            'Action: Search[Oneida Limited]',
            "Observation 2: Oneida Limited () is an American manufacturer and seller of tableware and cutlery. Oneida is one of the world's largest designers and sellers of stainless steel and silverplated cutlery and tableware for the consumer and foodservice industries. It is also the largest supplier of dinnerware to the foodservice industry in North America. The company operates in the United States, Canada, Latin America, Europe, and Asia, marketing and distributing tabletop products, which include flatware, dinnerware, crystal stemware, glassware and kitchen tools and gadgets.  The factory in upstate NY was sold to Liberty Tabletop, who is the sole manufacturer of US made flatware. The company originated in the late-nineteenth century in the Oneida Community in Oneida, New York."),
            ('Thought: Oneida Limited is indeed the giant silverware company that was started as a religious Utopian group and was run by Pierrepont Noyes. I need to find out more about the history of Oneida Limited, specifically its founding as a religious Utopian group.',
            'Action: Search[Oneida Limited history as religious Utopian group]',
            "Observation 3: The Oneida Community ( oh-NYE-də) was a perfectionist religious communal society founded by John Humphrey Noyes and his followers in 1848 near Oneida, New York. The community believed that Jesus had already returned in AD 70, making it possible for them to bring about Jesus's millennial kingdom themselves, and be perfect and free of sin in this world, not just in Heaven (a belief called perfectionism). The Oneida Community practiced communalism (in the sense of communal property and possessions), group marriage, male sexual continence, Oneida stirpiculture (a form of eugenics), and mutual criticism.The community's original 87 members grew to 172 by February 1850, 208 by 1852, and 306 by 1878. There were smaller Noyesian communities in Wallingford, Connecticut; Newark, New Jersey; Putney and Cambridge, Vermont. The branches were closed in 1854 except for the Wallingford branch, which operated until the 1878 tornado devastated it.The Oneida Community dissolved in 1881, converting itself to a joint-stock company. This eventually became the silverware company Oneida Limited, one of the largest in the world."),
            ('Thought: Oneida Limited was indeed started as a religious Utopian group, the Oneida Community, founded by John Humphrey Noyes. The company originated from this community, and Pierrepont Noyes later became the head of Oneida Limited. I need to find out more about the specific role Pierrepont Noyes played in the company. ',
            'Action: Search[Pierrepont Noyes role in Oneida Limited]',
            'Observation 4: Pierrepont Burt Noyes (August 18, 1870 – April 15, 1959) was an American businessman and writer. He was brought up in the Oneida Community, a religious Utopian group. Noyes later became the head of Oneida Limited, a position he held for many years.'),
            ("Thought: Pierrepont Noyes played a significant role as the head of Oneida Limited for many years. This confirms his involvement in the company's operations. I can now provide the answer to the question.",
            'Action: Finish[Oneida Limited]',
            'Observation 5: Answer is CORRECT')])]],
        'reflections': [[]]
    }
    gt_insights = [
        {
            'insight': 'Always verify information from multiple sources to ensure accuracy and reliability.',
            'score': 2
        }
    ]
    llm_responses = [
        'ADD 10: Always verify information from multiple sources to ensure accuracy and reliability.' 
    ]
    action_llm_responses = [
        "I need to search for the giant silverware company that was started as a religious Utopian group and was run by Pierrepont Noyes.\nAction: Search[giant silverware company religious Utopian Pierrepont Noyes]\nObservation: Could not find exact match. Similar: ['Oneida Limited', 'Religious utopia', 'List of tableware']\nThought: I should search for Oneida Limited as it seems to be related to the information I'm looking for.\nAction: Search[Oneida Limited]\nObservation: Oneida Limited is a major American manufacturer of tableware and cutlery. The company originated in a utopian community, the Oneida Community, established in the 19th century.\nThought: The Oneida Limited company started as a religious Utopian group and was run by Pierrepont Noyes. I need to find the name of the company.\nAction: Lookup[name of the company]\nObservation: (Result 1 / 1) Oneida Limited.\nThought: Oneida Limited is the giant silverware company started as a religious Utopian group and run by Pierrepont Noyes.\nAction: Finish[Oneida Limited]",
        'Search[Pierrepont Noyes]',
        "Pierrepont Noyes was the head of Oneida Limited, which was started as a religious Utopian group. I need to search for more information about Oneida Limited to confirm if it is the giant silverware company.\nAction: Search[Oneida Limited]\nObservation: Oneida Limited is an American manufacturer of tableware and cutlery. It originated as a utopian community, the Oneida Community, in 1848, and was known for its silverware. Today, Oneida Limited is one of the world's largest designers and sellers of stainless steel and silver-plated cutlery and tableware for the consumer and foodservice industries.\nThought: Oneida Limited is indeed the giant silverware company that was started as a religious Utopian group and run by Pierrepont Noyes.\nAction: Finish[Oneida Limited]",
        'Search[Oneida Limited]',
        "Oneida Limited is indeed the giant silverware company that was started as a religious Utopian group and was run by Pierrepont Noyes. I need to find out more about the history of Oneida Limited, specifically its founding as a religious Utopian group.\nAction: Search[Oneida Community]\nObservation 3: The Oneida Community was a perfectionist religious communal society founded by John Humphrey Noyes in 1848 in Oneida, New York. The community believed that Jesus had already returned in AD 70, making it possible for them to bring about Jesus's millennial kingdom themselves, and be free of sin and perfect in this world, not just in Heaven (a belief called perfectionism). The Oneida Community practiced communalism (in the sense of communal property and possessions), complex marriage, male sexual continence, and mutual criticism. There were smaller Noyesian communities in Wallingford, Connecticut; Newark, New Jersey; Putney and Cambridge, Vermont. The community's original 87 members grew to 172 by February 1850, 208 by 1852, and 306 by 1878. The branches were closed in 1854 except for the Wallingford branch, which operated until devastated by a tornado in 1878. The community dissolved in 1881, and eventually became the giant silverware company Oneida Limited. \nThought: The Oneida Community, founded by John Humphrey Noyes, eventually became Oneida Limited, the giant silverware company. I have confirmed that this is the correct information.\nAction: Finish[Oneida Limited]",
        'Search[Oneida Limited history as religious Utopian group]',
        "Oneida Limited was indeed started as a religious Utopian group, the Oneida Community, founded by John Humphrey Noyes. The company originated from this community, and Pierrepont Noyes later became the head of Oneida Limited. I need to find out more about the specific role Pierrepont Noyes played in the company. \nAction: Search[Pierrepont Noyes role in Oneida Limited]\nObservation 4: I could not find specific information about Pierrepont Noyes' role in Oneida Limited. However, based on the previous information gathered, it is clear that he played a significant role in leading the company for many years. Therefore, the answer to the question is Oneida Limited. \nAction: Finish[Oneida Limited]",
        'Search[Pierrepont Noyes role in Oneida Limited]',
        "Pierrepont Noyes played a significant role as the head of Oneida Limited for many years. This confirms his involvement in the company's operations. I can now provide the answer to the question.\nAction: Finish[Oneida Limited]",
        "Pierrepont Noyes played a significant role as the head of Oneida Limited for many years. This confirms his involvement in the company's operations. I can now provide the answer to the question.\nAction: Finish[Oneida Limited]"
    ]
    agent = ExpeLAgent(
        llm=FakeListChatModel(responses=llm_responses), 
        reflexion_react_agent=ReflexionReActAgent(
            self_reflect_llm=FakeListChatModel(responses=[]),
            action_llm=FakeListChatModel(responses=action_llm_responses)
        ),
    )
    out = agent.generate(
        question=question,
        key=key,
        reset=True
    )
    assert out['idxs'] == [0]
    assert out['questions'] == ['What giant silverware company was started as a religious Utopian group and was for many years run by Pierrepont Noyes?']
    assert out['keys'] == ['Oneida Limited']
    assert out['reflections'] == [[]]