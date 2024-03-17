"""Unit tests for ReAct."""
import yaml
import alfworld
import alfworld.agents.environment
from langchain.agents.react.base import DocstoreExplorer
from langchain.llms.fake import FakeListLLM
from langchain_community.chat_models.fake import FakeListChatModel
from langchain_core.language_models.chat_models import BaseChatModel
from tiktoken import Encoding

from discussion_agents.cog.prompts.react import (
    REACT_WEBTHINK_SIMPLE6_FEWSHOT_EXAMPLES,
    REACT_INSTRUCTION_HOTPOTQA,
    REACT_WEBTHINK_SIMPLE3_FEVER_EXAMPLES,
    REACT_INSTRUCTION_FEVER,
    REACT_ALFWORLD_PROMPTS_EXAMPLE,
    REACT_ALFWORLD_INSTRUCTION
)
from discussion_agents.cog.agent.react import ReActAgent, ZeroShotReActAgent
from tests.fixtures.agent import alfworld_env
from discussion_agents.cog.modules.memory.react import ReActMemory



def test_init() -> None:
    """Test initialization."""
    llm = FakeListChatModel(responses=["1"])
    agent = ReActAgent(llm=llm)
    assert isinstance(agent, ReActAgent)
    assert isinstance(agent.llm, BaseChatModel)
    assert isinstance(agent.memory, ReActMemory)
    assert agent.max_steps == 6
    assert agent.max_tokens == 3896
    assert isinstance(agent.docstore, DocstoreExplorer)
    assert isinstance(agent.enc, Encoding)

    assert agent._step_n == 1
    assert agent._finished == False
    assert agent.memory.scratchpad == ""


def test_generate() -> None:
    """Test generate."""
    q = 'Who was once considered the best kick boxer in the world, however he has been involved in a number of controversies relating to his "unsportsmanlike conducts" in the sport and crimes of violence outside of the ring'

    # Test a single max step and verify no more steps can be taken.
    responses = [
        ' I need to search for the best kick boxer in the world, and then find any controversies or crimes they have been involved in.\nAction: Search[best kick boxer in the world]\nObservation: (Result 1/1) Ramon Dekkers is considered by many to be the best kickboxer in the world.\nThought: It mentions "unsportsmanlike conducts" and crimes of violence. I need to find more information about Ramon Dekkers.\nAction: Lookup[crimes]\nObservation: (Result 1/1) Dekkers was involved in a number of controversies relating to his "unsportsmanlike conducts" in the sport and crimes of violence outside of the ring.\nThought: Ramon Dekkers has been involved in controversies and crimes. I need to find more information about them.\nAction: Lookup[controversies]\nObservation: (Result 1/1) Dekkers was known for his aggressive style and has been involved in a number of controversies, including a bar brawl and an altercation with a bouncer.\nThought: It mentions "unsportsmanlike conducts" and crimes of violence. I need to find more information about the controversies and crimes.\nAction: Lookup[unsportsmanlike conducts]\nObservation: (Result',
        ' Search[best kick boxer]\nObservation: The best kick boxer in the world is often a highly debated topic, but some notable names include Semmy Schilt, Peter Aerts, Ernesto Hoost, and Ramon Dekkers.\nThought: Since the question mentions controversies and crimes, I should focus on more recent kick boxers. I will look up the controversies and crimes of Semmy Schilt.\nAction: Lookup[controversies and crimes]\nObservation: (Result 1/1) Semmy Schilt has been involved in several controversies, including accusations of using performance-enhancing drugs and unsportsmanlike conducts such as eye-gouging and low blows.\nThought: The question mentions "unsportsmanlike conducts" specifically, so I will look up more information on those incidents.\nAction: Lookup[unsportsmanlike conducts]\nObservation: (Result 1/1) Semmy Schilt has been known for his aggressive and sometimes controversial fighting style, with incidents such as eye-gouging and low blows being reported by his opponents.\nThought: The question also mentions crimes outside of the ring, so I will search for any criminal record or charges against Semmy Schilt.\nAction: Search[Semmy Schilt criminal record]\nObservation',
    ]
    llm = FakeListChatModel(responses=responses)
    agent = ReActAgent(llm=llm)
    out = agent.generate(question=q, examples=REACT_WEBTHINK_SIMPLE6_FEWSHOT_EXAMPLES, prompt_template=REACT_INSTRUCTION_HOTPOTQA)
    assert isinstance(out, str)
    assert agent._step_n <= agent.max_steps
    assert not agent._finished

    # Verify no more steps can be taken.
    out = agent.generate(question=q, reset=False)
    assert not out
    assert isinstance(out, list)
    for triplet in out:
        assert isinstance(triplet, tuple)
    assert agent._step_n == agent.max_steps + 1
    assert not agent._finished

    scratchpad = "\n".join(agent.retrieve()["scratchpad"].split("\n")[:-1])
    assert scratchpad.strip() == gt_out

    # Test agent runs out of tokens (must ensure that max_steps is not reached and task is not finished).
    responses = [
        ' I need to search for the best kick boxer in the world, and then find any controversies or crimes they have been involved in.\nAction: Search[best kick boxer in the world]\nObservation: (Result 1/1) Ramon Dekkers is considered by many to be the best kickboxer in the world.\nThought: It mentions "unsportsmanlike conducts" and crimes of violence. I need to find more information about Ramon Dekkers.\nAction: Lookup[crimes]\nObservation: (Result 1/1) Dekkers was involved in a number of controversies relating to his "unsportsmanlike conducts" in the sport and crimes of violence outside of the ring.\nThought: Ramon Dekkers has been involved in controversies and crimes. I need to find more information about them.\nAction: Lookup[controversies]\nObservation: (Result 1/1) Dekkers was known for his aggressive style and has been involved in a number of controversies, including a bar brawl and an altercation with a bouncer.\nThought: It mentions "unsportsmanlike conducts" and crimes of violence. I need to find more information about the controversies and crimes.\nAction: Lookup[unsportsmanlike conducts]\nObservation: (Result',
        " INVALID[best kick boxer]\n",
    ]
    llm = FakeListChatModel(responses=responses)
    agent = ReActAgent(
        llm=llm, max_steps=3, max_tokens=1750
    )  # 3 steps leads to 1774 tokens.
    out = agent.generate(question=q)

    gt_out = (
        "Thought: I need to search for the best kick boxer in the world, and then find any controversies or crimes they have been involved in.\n"
        "Action: INVALID[best kick boxer]\n"
        "Observation 1: Invalid Action. Valid Actions are Lookup[<topic>] Search[<topic>] and Finish[<answer>].\n"
        "Thought: I need to search for the best kick boxer in the world, and then find any controversies or crimes they have been involved in.\n"
        "Action: INVALID[best kick boxer]\n"
        "Observation 2: Invalid Action. Valid Actions are Lookup[<topic>] Search[<topic>] and Finish[<answer>]."
    )

    assert isinstance(out, list)
    for triplet in out:
        assert isinstance(triplet, tuple)
    assert "\n".join(["\n".join(triplet) for triplet in out]) == gt_out
    assert agent.memory.load_memories()["scratchpad"].strip() == gt_out

    # Test full trajectoy/trial till finish.
    responses = [
        ' I need to search for the best kick boxer in the world, and then find any controversies or crimes they have been involved in.\nAction: Search[best kick boxer in the world]\nObservation: (Result 1/1) Ramon Dekkers is considered by many to be the best kickboxer in the world.\nThought: It mentions "unsportsmanlike conducts" and crimes of violence. I need to find more information about Ramon Dekkers.\nAction: Lookup[crimes]\nObservation: (Result 1/1) Dekkers was involved in a number of controversies relating to his "unsportsmanlike conducts" in the sport and crimes of violence outside of the ring.\nThought: Ramon Dekkers has been involved in controversies and crimes. I need to find more information about them.\nAction: Lookup[controversies]\nObservation: (Result 1/1) Dekkers was known for his aggressive style and has been involved in a number of controversies, including a bar brawl and an altercation with a bouncer.\nThought: It mentions "unsportsmanlike conducts" and crimes of violence. I need to find more information about the controversies and crimes.\nAction: Lookup[unsportsmanlike conducts]\nObservation: (Result',
        " Finish[Badr Hari]\n",
    ]
    llm = FakeListChatModel(responses=responses)
    agent = ReActAgent(llm=llm, max_steps=5)
    out = agent.generate(question=q)
    gt_out = (
        "\n"
        "Thought: I need to search for the best kick boxer in the world, and then find any controversies or crimes they have been involved in.\n"
        "Action: Finish[Badr Hari]\n"
        "Observation 1: Badr Hari"
    )
    assert isinstance(out, list)
    for triplet in out:
        assert isinstance(triplet, tuple)
    assert gt_out == gt_out
    assert agent.memory.load_memories()["scratchpad"] == gt_out

def test_reset(react_agent: ReActAgent) -> None:
    """Test reset."""
    assert react_agent.memory.scratchpad == ""
    react_agent.memory.scratchpad = "abc"
    assert not react_agent._finished
    react_agent._finished = True
    assert react_agent._step_n == 1
    react_agent._step_n = 10
    react_agent.reset()
    assert react_agent.memory.scratchpad == ""
    assert not react_agent._finished
    assert react_agent._step_n == 1


def test_retrieve(react_agent: ReActAgent) -> None:
    """Test retrieve."""
    out = react_agent.retrieve()
    assert isinstance(out, dict)
    assert "scratchpad" in out
    assert not out["scratchpad"]


def test_zeroshot_react_init() -> None:
    """Tests ZeroShotReActAgent's initialization."""
    agent = ZeroShotReActAgent(llm=FakeListLLM(responses=["1"]))
    assert agent is not None
    assert agent.llm is not None
    assert len(agent.tools) >= 1
    assert agent.agent is not None


def test_FEVER_react_generate() -> None:

    """Test generate."""
    q = 'Brad Wilk died before being a drummer for Greta.'

    # Test simple run.
    responses = [
        'Thought: I need to search for Brad Wilk and see if he died before being a drummer for Greta.\nAction: Search[Brad Wilk]\\nObservation 1: Could not find [Brad Wilk]. Similar: [\'Brad Wilk\', \'Rage Against the Machine\', \'Audioslave\', \'Prophets of Rage\', \'Wilk\', \'Tim Commerford\', \'Tom Morello\', \'Show Me How to Live (song)\', \'Greta (band)\', \'Zack de la Rocha\']\nThought: The search did not return Brad Wilk directly, but it mentioned Greta (band). I should search for Greta (band) to see if Brad Wilk was a drummer for them.\nAction: Search[Greta (band)]\nObservation 2: Greta was an American hard rock band formed in 1990 by Paul Plagens, Kyle Baer, Josh Gordon, and Brad Wilk.Wilk left the band in 1991 to join Rage Against the Machine, and was replaced on the drums by Scott Carneghi who went on to co-found the band Buffalocomotive in 2012.The band signed a two-record deal with Mercury Records in 1993 and released their debut album, No Biting, on September 21 of that year. Their second and final album, This Is Greta, was released in 1995, after which the band was dropped from the label. Greta disbanded in 1995. Vocalist Paul Plagens died in 2015\nThought: The observation clearly states that Brad Wilk left Greta in 1991 to join Rage Against the Machine, so he did not die before being a drummer for Greta.\nAction: Finish[REFUTES]\nObservation 3: REFUTES' 
    ]
    llm = FakeListChatModel(responses=responses)
    agent = ReActAgent(llm=llm)
    out = agent.generate(question=q, examples=REACT_WEBTHINK_SIMPLE3_FEVER_EXAMPLES, prompt_template=REACT_INSTRUCTION_FEVER)
    assert isinstance(out, str)
    assert agent._step_n <= agent.max_steps + 1
    assert not agent._finished


def test_Alfworld_react_generate(alfworld_env) -> None:
    """Testing for Alfworld."""
    config = alfworld_env
    split = "eval_out_of_distribution"
    env = getattr(alfworld.agents.environment, config["env"]["type"])(config, train_eval=split)
    env = env.init_env(batch_size=1)
    ob, info = env.reset()
    ob = '\n'.join(ob[0].split('\n\n')[1:])
    name = '/'.join(info['extra.gamefile'][0].split('/')[-3:-1])
    d = REACT_ALFWORLD_PROMPTS_EXAMPLE
    prefixes = {
        'pick_and_place': 'put',
        'pick_clean_then_place': 'clean',
        'pick_heat_then_place': 'heat',
        'pick_cool_then_place': 'cool',
        'look_at_obj': 'examine',
        'pick_two_obj': 'puttwo'
    }
    for i, (k, v) in enumerate(prefixes.items()):
        if name.startswith(k):
            prompt = 'Interact with a household to solve a task. Here are two examples.\n' + d[f'react_{v}_1'] + d[f'react_{v}_0'] + '\nHere is the task.\n'

    response = [
        '''
        Action 1: think: To solve the task, I need to find a tomato, then cool it with the fridge, and finally put it in the microwave.\nObservation 1:  OK.\nAction 2: think: First, I need to find a tomato. A tomato is more likely to be found in countertop (1-3), fridge (1), cabinet (1-6), shelf (1-3), and drawer (1-3). I can start by checking countertop 1.\nObservation 2:  OK.\nAction 3: go to countertop 1\nObservation 3:  On the countertop 1, you see a dishsponge 1, and a mug 2.\nAction 4: go to countertop 2\nObservation 4:  On the countertop 2, you see a butterknife 1, a cellphone 1, a creditcard 1, a knife 1, a lettuce 1, a saltshaker 2, a saltshaker 1, a statue 1, and a tomato 1.\nAction 5: take tomato 1 from countertop 2\nObservation 5:  You pick up the tomato 1 from the countertop 2.\nAction 6: think: Now that I have the tomato, I need to cool it with the fridge.\nObservation 6:  OK.\nAction 7: go to fridge 1\nObservation 7:  The fridge 1 is closed.\nAction 8: cool tomato 1 with fridge 1\nObservation 8:  You cool the tomato 1 using the fridge 1.\nAction 9: think: Now the tomato is cool. Next, I need to put it in the microwave.\nObservation 9:  OK.\nAction 10: go to microwave 1\nObservation 10:  The microwave 1 is closed.\nAction 11: put tomato 1 in/on microwave 1\nObservation 11:  Nothing happens.\nAction 12: open microwave 1\nObservation 12:  You open the microwave 1. The microwave 1 is open. In it, you see a egg 1.\nAction 13: put tomato 1 in/on microwave 1\nObservation 13:  You put the tomato 1 in/on the microwave 1.
        '''    
    ]

    llm = FakeListChatModel(responses=response)
    agent = ReActAgent(llm=llm)
    agent.set_Alfworld()
    out = agent.step(question=ob, examples=prompt, prompt_template=REACT_ALFWORLD_INSTRUCTION,env_output=None)
    out = out[0].split(':')[-1].strip()
    assert out == 'You put the tomato 1 in/on the microwave 1.'
