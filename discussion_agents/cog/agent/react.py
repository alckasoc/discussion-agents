"""ReAct Agent implementation and LangChain's zero-shot ReAct.

This includes the original ReAct agent implementation and the LangChain-adapted
Zero-shot ReAct, with a wikipedia searcher default tool.

Original Paper: https://arxiv.org/abs/2210.03629
Paper Repository: https://github.com/ysymyth/ReAct
LangChain: https://github.com/langchain-ai/langchain
LangChain ReAct: https://python.langchain.com/docs/modules/agents/agent_types/react
"""
from typing import Any, Dict, List, Optional, Tuple

import tiktoken

from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain.agents.react.base import DocstoreExplorer
from langchain_community.docstore.wikipedia import Wikipedia
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool, tool
from tiktoken.core import Encoding



from discussion_agents.cog.agent.base import BaseAgent
from discussion_agents.cog.functional.react import _is_halted, _prompt_agent
from discussion_agents.cog.modules.memory.react import ReActMemory
from discussion_agents.utils.parse import parse_action, remove_newline

from discussion_agents.cog.prompts.react import (
    REACT_INSTRUCTION_HOTPOTQA,
    REACT_INSTRUCTION_FEVER,
    REACT_ALFWORLD_INSTRUCTION
)

# is_think: bool
# is_think: bool = True
# .generate(question=q, is_think=False)




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
        max_steps: int = 6,
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
        self.is_think = True # this is for the 


    def set_Alfworld(self) -> None:
        """
        Set the 'is_think' attribute of the object to False.
        
        This method sets the 'is_think' attribute of the object to False,
        indicating that the object is not currently in a thinking state.
        
        Returns:
            None
        """
        self.is_think = False
        return
    
    def step(self, question: str, examples: str, prompt_template: str, env_output: Optional[str] = None) -> List[str]:
        """
        Perform a step in the conversation based on the given question and examples.

        This method processes a step in the conversation, which includes:
        - Handling environment output if provided.
        - Generating an action based on the provided question, examples, and prompt template.
        - Parsing the action and performing the corresponding observation.
        Args:
            question (str): The question for the conversation step.
            examples (str): Examples relevant to the question.
            prompt_template (str): Template for generating prompts.
            env_output (Optional[str], optional): Output from the environment. Defaults to None.

        Returns:
            Tuple[List, bool]: A tuple containing a list of outputs from the step and a boolean
            indicating if the conversation is finished.
        """

        out = []

        # Handling environment output if provided
        if env_output:
            self.memory.add_memories(f"\nObservation {self._step_n}: ")
            self.memory.add_memories(env_output)

        # Handling "Thinking" mode
        if self.is_think:
            self.memory.add_memories("\nThought:")
            thought = _prompt_agent(
                llm=self.llm,
                question=question,
                scratchpad=self.memory.load_memories()["scratchpad"],
                examples=examples,
                prompt_template=prompt_template
            ).strip()
            self.memory.add_memories(" " + thought)
            out.append(thought)

        self.memory.add_memories(f"\nAction {self._step_n}:")

        # Generating an action based on the question, examples, and prompt template
        action = _prompt_agent(
            llm=self.llm,
            question=question,
            scratchpad=self.memory.load_memories()["scratchpad"],
            examples=examples,
            prompt_template=prompt_template,
            stop=''.join(['\\n'])
        ).strip()

        # Processing the action if not in "Thinking" mode
        if not self.is_think:
            if action.startswith('>'):
                action = action.replace('>', '')
            if action.startswith('Action'):
                action = action.split(':', 1)[1]
            if not action.startswith('think'):
                action = action.replace(' in ', ' in/on ').strip()

        self.memory.add_memories(" " + action)
        out.append(action)

        # Processing based on "Thinking" mode
        if not self.is_think:
            self._step_n += 1
            return out
        else:
            self.memory.add_memories(f"\nObservation {self._step_n}: ")

            action_type, query = parse_action(action)

            # Handling different action types
            if action_type.lower() == "finish":
                self._answer = query
                self._finished = True
                self.memory.add_memories(query)

            elif action_type.lower() == "search":
                try:
                    obs = remove_newline(self.docstore.search(query))
                except Exception:
                    self.memory.add_memories(
                        "Could not find that page, please try again."
                    )

            elif action_type.lower() == "lookup":
                try:
                    obs = remove_newline(self.docstore.lookup(query))
                except ValueError:
                    obs = "The last page Searched was not found, so you cannot Lookup a keyword in it. Please try one of the similar pages given."
            else:
                obs = "Invalid Action. Valid Actions are Lookup[<topic>] Search[<topic>] and Finish[<answer>]."
            self.memory.add_memories(obs)

            out.append(
                "\n".join(
                    (
                        f"Thought: {thought}",
                        f"Action: {action}",
                        f"Observation {self._step_n}: {obs}"
                    )
                )
           )



            out.append(self.memory.load_memories()["scratchpad"].split("\n")[-1])

        self._step_n += 1

        # Checking if conversation is finished
        finished = _is_halted(
            finished=self._finished,
            step_n=self._step_n,
            max_steps=self.max_steps,
            question=question,
            scratchpad=self.memory.load_memories()["scratchpad"],
            max_tokens=self.max_tokens,
            enc=self.enc,
            examples=examples,
            prompt_template=prompt_template
        )

        if finished:
            self.is_think = True

        return out

    def generate(self, question: str, reset: bool = True, examples: str = "", env: Any = None, prompt_template: str = "") -> str:
        """
        Processes a given question through ReAct.

        Iteratively applies the think-act-observe cycle to generate an answer for the question.
        The process continues until the operation is halted based on certain conditions.

        Args:
            question (str): The question to be processed.
            reset (bool, optional): Whether to reset the internal state before processing. Defaults to True.
            prompt_template (str): The template for benchmark
            examples (str): The example of text generated.
            env (Alfworld_environment): The variable to interact with Alfworld.

        Returns:
            str: The accumulated output from the ReAct process.
        """

        if not reset:
            # No need to reset
            pass
        else:
            self.reset()

        out = ""

        while True:
            is_halted = _is_halted(
                finished=self._finished,
                step_n=self._step_n,
                max_steps=self.max_steps,
                question=question,
                scratchpad=self.memory.load_memories()["scratchpad"],
                max_tokens=self.max_tokens,
                enc=self.enc,
                examples=examples,
                prompt_template=prompt_template
            )
            if is_halted:
                break

            thought = self.think(question, self.memory.load_memories()["scratchpad"], examples, prompt_template)
            out += "\n" + thought

            action = self.act(question, self.memory.load_memories()["scratchpad"], examples, prompt_template)
            out += "\n" + action

            self.observe(action)
            out += "\n" + self.memory.load_memories()["scratchpad"].split("\n")[-1]

            self._step_n += 1

        return out


    def think(self, question: str, scratchpad: str, examples: str, prompt_template: str) -> str:
        """
        Generates a thought based on the given question, scratchpad, examples, and prompt template.

        Args:
            question (str): The question to be processed.
            scratchpad (str): The current state of the scratchpad.
            examples (str): The example of text generated.
            prompt_template (str): The template for the prompt.

        Returns:
            str: The generated thought.
        """
        self.memory.add_memories("\nThought:")
        thought = _prompt_agent(
            llm=self.llm,
            question=question,
            scratchpad=scratchpad,
            examples=examples,
            prompt_template=prompt_template,
            stop=''.join(['\\n'])
        ).strip()
        self.memory.add_memories(" " + thought)
        return thought


    def act(self, question: str, scratchpad: str, examples: str, prompt_template: str) -> str:
        """
        Generates an action based on the given question, scratchpad, examples, and prompt template.

        Args:
            question (str): The question to be processed.
            scratchpad (str): The current state of the scratchpad.
            examples (str): The example of text generated.
            prompt_template (str): The template for the prompt.

        Returns:
            str: The generated action.
        """
        self.memory.add_memories(f"\nAction {self._step_n}:")
        action = _prompt_agent(
            llm=self.llm,
            question=question,
            scratchpad=scratchpad,
            examples=examples,
            prompt_template=prompt_template,
            stop=''.join(['\\n'])
        ).strip()
        self.memory.add_memories(" " + action)
        return action


    def observe(self, action: str) -> None:
        """
        Observes the given action and performs the corresponding operation.

        Args:
            action (str): The action to be observed.

        Raises:
            None
        """
        action_type, query = parse_action(action)

        if action_type.lower() == "finish":
            self._answer, self._finished = query, True
            self.memory.add_memories(query)
        elif action_type.lower() == "search":
            search_result = self.docstore.search(query) or "Could not find that page, please try again."
            self.memory.add_memories(remove_newline(search_result))
        elif action_type.lower() == "lookup":
            lookup_result = (
                self.docstore.lookup(query)
                if self.docstore.last_search_result
                else "No previous search result available to perform a lookup. Please search for a page first."
            )
            self.memory.add_memories(
                remove_newline(lookup_result)
                if isinstance(lookup_result, str)
                else "The last page Searched was not found, so you cannot Lookup a keyword in it. Please try one of the similar pages given."
            )
        else:
            self.memory.add_memories("Invalid Action. Valid Actions are Lookup[<topic>] Search[<topic>] and Finish[<answer>].")
        return 
        # Think, Act, and Observe steps...

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



@tool
def search(query: str) -> str:
    """Searches Wikipedia given query."""
    docstore = DocstoreExplorer(Wikipedia())
    return docstore.search(query)



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
