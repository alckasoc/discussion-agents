"""ReAct Agent implementation and LangChain's zero-shot ReAct.

This includes the original ReAct agent implementation and the LangChain-adapted
Zero-shot ReAct, with a wikipedia searcher default tool.

Original Paper: https://arxiv.org/abs/2210.03629
Paper Repository: https://github.com/ysymyth/ReAct
LangChain: https://github.com/langchain-ai/langchain
LangChain ReAct: https://python.langchain.com/docs/modules/agents/agent_types/react
"""
from typing import Any, Dict, List, Optional

import tiktoken

from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain.agents.react.base import DocstoreExplorer
from langchain_community.docstore.wikipedia import Wikipedia
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool, tool
from tiktoken.core import Encoding

import yaml
import alfworld
import alfworld.agents.environment


from discussion_agents.cog.agent.base import BaseAgent
from discussion_agents.cog.functional.react import _is_halted, _prompt_agent
from discussion_agents.cog.modules.memory.react import ReActMemory
from discussion_agents.utils.parse import parse_action, remove_newline

from discussion_agents.cog.prompts.react import (
    REACT_INSTRUCTION_HOTPOTQA,
    REACT_INSTRUCTION_FEVER,
    REACT_WEBTHINK_SIMPLE6_FEWSHOT_EXAMPLES,
    REACT_WEBTHINK_SIMPLE3_FEVER_EXAMPLES,
    REACT_ALFWORLD_PROMPTS_EXAMPLE,
    REACT_ALFWORLD_INSTRUCTION
)

import yaml
import alfworld
import alfworld.agents.environment




ALFWORLD = 'alfworld'
HOTPOTQA = 'hotpotqa'
FEVER = 'fever'



class ReActAgent(BaseAgent):
    """ReAct agent from the original paper.

    Implements the ReAct algorithm as described in the original paper.
    This agent uses a language model to iteratively process a question
    through a sequence of think-act-observe steps, utilizing a document
    store for information retrieval.

    Attributes:
        llm (BaseChatModel): The language model used by the agent.
        max_steps (int): Maximum number of steps to process the question.
        max_tokens (int): Maximum token limit for the language model.
        docstore (DocstoreExplorer): Document store for information retrieval.
        enc (Encoding): Encoder for calculating token lengths.
        benchmark_type (str): Specifies the benchmark type used for selecting the appropriate examples. Acceptable values are limited to 'HotpotQA' or 'FEVER' or 'Alfworld'.

    See: https://github.com/ysymyth/ReAct
    """

    def __init__(
        self,
        llm: BaseChatModel,
        memory: Optional[ReActMemory] = None,
        max_steps: int = 20,
        max_tokens: int = 3896,
        docstore: DocstoreExplorer = DocstoreExplorer(Wikipedia()),
        enc: Encoding = tiktoken.encoding_for_model("gpt-3.5-turbo"),
    ) -> None:
        """Initialization."""
        super().__init__()
        self.llm = llm

        if not memory:
            self.memory = ReActMemory()
        else:
            self.memory = memory

        self.max_steps = max_steps
        self.max_tokens = max_tokens
        self.docstore = docstore
        self.enc = enc



        # Internal variables.
        self._step_n = 1  #: :meta private:
        self._finished = False  #: :meta private:



    def generate(self, question: str, reset: bool = True, examples: str = None , env: Any = None) -> str:
        """Processes a given question through ReAct.

        Iteratively applies the think-act-observe cycle to generate an answer for the question.
        The process continues until the operation is halted based on certain conditions.

        Args:
            question (str): The question to be processed.
            reset (bool, optional): Whether to reset the internal state before processing. Defaults to True.

        Returns:
            str: The accumulated output from the ReAct process.
        """

        input_check = examples
        benchmark_type = check_type(examples=input_check)


        if benchmark_type == HOTPOTQA:
            instruction = REACT_INSTRUCTION_HOTPOTQA
        elif benchmark_type == FEVER:
            instruction = REACT_INSTRUCTION_FEVER
        elif benchmark_type == ALFWORLD:
            instruction = REACT_ALFWORLD_INSTRUCTION

        if reset:
            self.reset()

        out = ""
        while not _is_halted(
            finished=self._finished,
            step_n=self._step_n,
            max_steps=self.max_steps,
            question=question,
            scratchpad=self.memory.load_memories()["scratchpad"],
            max_tokens=self.max_tokens,
            enc=self.enc,
            examples=examples,
            instruction=instruction
        ):
            
            # Think.
            if benchmark_type == HOTPOTQA or benchmark_type == FEVER:
                self.memory.add_memories("\nThought:")
                thought = _prompt_agent(
                    llm=self.llm,
                    question=question,
                    scratchpad=self.memory.load_memories()["scratchpad"],
                    examples=examples,
                    instruction=instruction
                    
                )
                self.memory.add_memories(" " + thought)
                out += "\n" + self.memory.load_memories()["scratchpad"].split("\n")[-1]

            # Act.
        
            self.memory.add_memories(f"\nAction {self._step_n}:")
            
            action = _prompt_agent(
                llm=self.llm,
                question=question,
                scratchpad=self.memory.load_memories()["scratchpad"],
                examples=examples,
                instruction=instruction
            ).strip()
            action = action.replace('>','').strip()
            self.memory.add_memories(" " + action)
            out += "\n" + self.memory.load_memories()["scratchpad"].split("\n")[-1]
            # Observe.
            self.memory.add_memories(f"\nObservation {self._step_n}: ")
            if benchmark_type == ALFWORLD:
                action = action.replace(' in ',' in\on ')
                observation, reward, done, info = env.step([action])
                observation, reward, done = process_ob(observation[0]), info['won'][0], done[0]

                if done :
                    print(1)
                    self._finished = True


                if 'think:' in action:
                    observation = 'OK.'
                self.memory.add_memories(" " + observation)
                

            else:
                action_type, query = parse_action(action)
                if action_type.lower() == "finish":
                    self._answer = query
                    self._finished = True
                    self.memory.add_memories(query)
                elif action_type.lower() == "search":
                    try:
                        self.memory.add_memories(
                            remove_newline(self.docstore.search(query))
                        )
                    except Exception:
                        self.memory.add_memories(
                            "Could not find that page, please try again."
                        )       
                elif action_type.lower() == "lookup":
                    try:
                        self.memory.add_memories(
                            remove_newline(self.docstore.lookup(query))
                        )
                    except ValueError:
                        self.memory.add_memories(
                            "The last page Searched was not found, so you cannot Lookup a keyword in it. Please try one of the similar pages given."
                        )
                else:
                    self.memory.add_memories(
                        "Invalid Action. Valid Actions are Lookup[<topic>] Search[<topic>] and Finish[<answer>]."
                    )
            self._step_n += 1
            out += "\n" + self.memory.load_memories()["scratchpad"].split("\n")[-1]
        return out

    def retrieve(self) -> Dict[str, Any]:
        """Retrieves the current state of the agent's memory.

        Returns:
            Dict[str, Any]: The current state of the agent's memory.
        """
        return self.memory.load_memories()

    def reset(self) -> None:
        """Resets the internal state of the ReAct agent.

        Sets the step number, finished flag, and scratchpad to their initial values.
        """
        self._step_n = 1
        self._finished = False
        self.memory.clear()
        return


@tool
def search(query: str) -> str:
    """Searches Wikipedia given query."""
    docstore = DocstoreExplorer(Wikipedia())
    return docstore.search(query)


def check_type(examples: str = None) -> str:

    lines = examples.split('\n')
    checkword = lines[0].split()[0]
    if checkword == 'Question:':
        return HOTPOTQA
    else:
        checkword = lines[3].split()[0]
        if checkword == 'Claim:':
            return FEVER
        line = lines[2]
        if 'Your task is to:' in line :
            return ALFWORLD
    return ValueError('Wrong Examples')


def process_ob(ob):
    if ob.startswith('You arrive at loc '):
        ob = ob[ob.find('. ')+2:]    
    return ob

class ZeroShotReActAgent(BaseAgent):
    """The Zero-Shot ReAct Agent class adapted from LangChain.

    Attributes:
        llm (Any): An attribute for a language model or a similar interface. The exact type is to be determined.
        tools (List[BaseTool]): A list of tools that the agent can use to interact or perform tasks.
        prompt (str, optional): An initial prompt for the agent. If not provided, a default prompt is fetched from a specified hub.

    See: https://github.com/langchain-ai/langchain/tree/master/libs/langchain/langchain/agents/react
    """

    def __init__(
        self,
        llm: Any,
        tools: List[BaseTool] = [],
        prompt: Optional[str] = None,
    ) -> None:
        """Initialization."""
        super().__init__()
        self.llm = llm  # TODO: Why is `LLM` not usable here?
        self.tools = tools
        self.tools.append(search)  # type: ignore
        prompt = hub.pull("hwchase17/react") if not prompt else prompt
        self.prompt = prompt
        if self.llm and self.tools and self.prompt:
            agent = create_react_agent(llm, tools, prompt)  # type: ignore
            agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)  # type: ignore
            self.agent = agent_executor

    def generate(self, observation_dict: Dict[str, str]) -> str:
        """Generates a response based on the provided observation dictionary.

        This method wraps around the `AgentExecutor`'s `invoke` method.

        Args:
            observation_dict (Dict[str, str]): A dictionary containing observation data.

        Returns:
            str: The generated response.
        """
        return self.agent.invoke(observation_dict)  # type: ignore






    