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
from tiktoken.core import Encoding

from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.docstore.wikipedia import Wikipedia
from langchain.agents.react.base import DocstoreExplorer
from langchain_core.tools import BaseTool, tool

from discussion_agents.cog.agent.base import BaseAgent
from discussion_agents.utils.parse import parse_action
from discussion_agents.cog.functional.react import _is_halted, react_think, react_act, react_observe

class ReActAgent(BaseAgent):
    """ReAct agent from the original paper.

    This agent has 2 methods: `search` and `generate`. It does not
    have any memory, planning, reflecting, or scoring capabilities.
    Given a question, this agent, equipped with Wikipedia search,
    attempts to answer the question in, a maximum of, 7 steps. Each step
    is a thought-action-observation sequence.

    Available actions are:
        - Search[], search for relevant info on Wikipedia (5 sentences)
        - Lookup[], lookup keywords in Wikipedia search
        - Finish[], finish task

    Note:
        By default, HOTPOTQA_FEWSHOT_EXAMPLES are used as fewshot context examples.
        You have the option to provide your own fewshot examples in the `generate` method.

    Attributes:
        llm (LLM): An instance of a language model used for processing and generating content.

    See: https://github.com/ysymyth/ReAct
    """
    def __init__(
        self, 
        llm: Any, 
        max_steps: int = 6, 
        max_tokens: int = 3896,
        docstore: Optional[DocstoreExplorer] = DocstoreExplorer(Wikipedia()),
        enc: Optional[Encoding] = tiktoken.encoding_for_model("gpt-3.5-turbo")
    ) -> None:
        """Initialization."""
        super().__init__()
        self.llm = llm
        self.max_steps = max_steps
        self.max_tokens = max_tokens
        self.docstore = docstore
        self.enc = enc

        # Internal variables.
        self.__step_n = 1  #: :meta private:
        self.__finished = False  #: :meta private:
        self.__scratchpad: str = ""  #: :meta private:

    def generate(self, question: str, reset: bool = True) -> str:
        if reset:
            self.reset()
        
        out = ""
        while not _is_halted(
            finished=self.__finished,
            step_n=self.__step_n, 
            max_steps=self.max_steps, 
            question=question, 
            scratchpad=self.__scratchpad, 
            max_tokens=self.max_tokens, 
            enc=self.enc
        ):
            # Think.
            self.__scratchpad = react_think(
                llm=self.llm, 
                question=question, 
                scratchpad=self.__scratchpad
            )
            out += "\n" + self.__scratchpad.split('\n')[-1]
            
            # Act.
            self.__scratchpad, action = react_act(
                llm=self.llm, 
                question=question, 
                scratchpad=self.__scratchpad
            )
            action_type, query = parse_action(action)
            out += "\n" + self.__scratchpad.split('\n')[-1]

            # Observe.
            observation = react_observe(
                action_type=action_type, 
                query=query, 
                scratchpad=self.__scratchpad, 
                step_n=self.__step_n, 
                docstore=self.docstore
            )
            self.__scratchpad = observation["scratchpad"]
            self.__step_n = observation["step_n"]
            self.__finished = observation["finished"]
            out += "\n" + self.__scratchpad.split('\n')[-1]

        return out


    def reset(self) -> None:
        self.__step_n = 1
        self.__finished = False
        self.__scratchpad: str = ""

# class ReActAgent(BaseAgent):
#     """ReAct agent from the original paper.

#     This agent has 2 methods: `search` and `generate`. It does not
#     have any memory, planning, reflecting, or scoring capabilities.
#     Given a question, this agent, equipped with Wikipedia search,
#     attempts to answer the question in, a maximum of, 7 steps. Each step
#     is a thought-action-observation sequence.

#     Available actions are:
#         - Search[], search for relevant info on Wikipedia (5 sentences)
#         - Lookup[], lookup keywords in Wikipedia search
#         - Finish[], finish task

#     Note:
#         By default, HOTPOTQA_FEWSHOT_EXAMPLES are used as fewshot context examples.
#         You have the option to provide your own fewshot examples in the `generate` method.

#     Attributes:
#         llm (LLM): An instance of a language model used for processing and generating content.

#     See: https://github.com/ysymyth/ReAct
#     """

#     llm: Any  # TODO: Why is `LLM` not usable here?

#     page: str = ""  #: :meta private:
#     result_titles: list = []  #: :meta private:
#     lookup_keyword: str = ""  #: :meta private:
#     lookup_list: list = []  #: :meta private:
#     lookup_cnt: int = 0  #: :meta private:

#     def search(self, entity: str, k: Optional[int] = 5) -> str:
#         """Performs a search operation for a given entity on Wikipedia.

#         It parses the search results and either returns a list of similar topics
#         (if the exact entity is not found) or the content of the Wikipedia page related to the entity.

#         Args:
#             entity (str): The entity to be searched for.
#             k (Optional[int]): An optional argument to specify the number of sentences to be returned
#                 from the Wikipedia page content.

#         Returns:
#             str: A string containing either the Wikipedia page content (trimmed to 'k' sentences) or
#                  a list of similar topics if the exact match is not found.
#         """
#         entity_ = entity.replace(" ", "+")
#         search_url = f"https://en.wikipedia.org/w/index.php?search={entity_}"
#         response_text = requests.get(search_url).text
#         soup = BeautifulSoup(response_text, features="html.parser")
#         result_divs = soup.find_all("div", {"class": "mw-search-result-heading"})
#         if result_divs:  # Mismatch.
#             self.result_titles = [
#                 clean_str(div.get_text().strip()) for div in result_divs
#             ]
#             obs = f"Could not find {entity}. Similar: {self.result_titles[:5]}."
#         else:
#             page = [
#                 p.get_text().strip() for p in soup.find_all("p") + soup.find_all("ul")
#             ]
#             if any("may refer to:" in p for p in page):
#                 obs = self.search("[" + entity + "]")
#             else:
#                 self.page = ""
#                 for p in page:
#                     if len(p.split(" ")) > 2:
#                         self.page += clean_str(p)
#                         if not p.endswith("\n"):
#                             self.page += "\n"
#                 obs = get_page_obs(self.page, k=k)

#                 # Reset lookup attributes.
#                 self.lookup_keyword = ""
#                 self.lookup_list = []
#                 self.lookup_cnt = 0

#         return obs

#     def generate(
#         self,
#         observation: str,
#         fewshot_examples: Optional[str] = WEBTHINK_SIMPLE6,
#     ) -> str:
#         """It takes an observation/question as input and generates a multi-step reasoning process.

#         The method involves generating thoughts and corresponding actions based on the observation,
#         and executing those actions which may include web searches, lookups, or concluding the reasoning process.

#         Args:
#             observation (str): The observation based on which the reasoning process is to be performed.
#             fewshot_examples (Optional[str]): A string containing few-shot examples to guide the language model.
#                                     Defaults to WEBTHINK_SIMPLE6.

#         Returns:
#             str: A string representing the entire reasoning process, including thoughts, actions, and observations
#                  at each step, culminating in a final answer or conclusion.
#         """
#         prompt_template = [
#             INSTRUCTION,
#             fewshot_examples,
#             "\n",
#             "Question: ",
#             "{observation}",
#             "\n",
#             "Thought {i}: ",
#         ]

#         # TODO: Find a way to enforce llm outputs.
#         done = False
#         out = ""
#         for i in range(1, 8):
#             # Create and run prompt.
#             prompt = PromptTemplate.from_template(
#                 "".join(prompt_template)  # type: ignore
#                 if not out
#                 else "".join(prompt_template[:-1]) + out  # type: ignore
#             )
#             chain = LLMChain(llm=self.llm, prompt=prompt)
#             thought_action = chain.run(observation=observation, i=i).split(
#                 f"\nObservation {i}:"
#             )[0]

#             # Get thought and action.
#             try:
#                 thought, action = thought_action.strip().split(f"\nAction {i}: ")
#                 thought = thought.split(f"Thought {i}: ")[-1]
#             except:
#                 thought = thought_action.strip().split("\n")[0]
#                 revised_prompt_template = (
#                     (
#                         "".join(prompt_template)  # type: ignore
#                         if not out
#                         else "".join(prompt_template[:-1]) + out  # type: ignore
#                     )
#                     + f"{thought}\n"
#                     + "Action {i}: "
#                 )
#                 revised_prompt = PromptTemplate.from_template(revised_prompt_template)
#                 chain = LLMChain(llm=self.llm, prompt=revised_prompt)
#                 action = chain.run(observation=observation, i=i).strip().split("\n")[0]

#             # Execute action and get observation.
#             if action.lower().startswith("search[") and action.endswith("]"):
#                 query = action[len("search[") : -1].lower()
#                 obs = self.search(query)
#                 if not obs.endswith("\n"):
#                     obs = obs + "\n"
#             elif action.lower().startswith("lookup[") and action.endswith("]"):
#                 keyword = action[len("lookup[") : -1].lower()

#                 # Reset lookup.
#                 if self.lookup_keyword != keyword:
#                     self.lookup_keyword = keyword
#                     self.lookup_list = construct_lookup_list(keyword, page=self.page)
#                     self.lookup_cnt = 0

#                 # All lookups used.
#                 if self.lookup_cnt >= len(self.lookup_list):
#                     obs = "No more results.\n"
#                 else:
#                     obs = (
#                         f"(Result {self.lookup_cnt + 1} / {len(self.lookup_list)}) "
#                         + self.lookup_list[self.lookup_cnt]
#                     )
#                     self.lookup_cnt += 1
#             elif action.lower().startswith("finish[") and action.endswith("]"):
#                 answer = action[len("finish[") : -1].lower()
#                 done = True
#                 obs = f"Episode finished. Answer: {answer}\n"
#             else:
#                 obs = "Invalid action: {}".format(action)

#             # Update out.
#             obs = obs.replace("\\n", "")
#             out += (
#                 f"Thought {i}: {thought}\n"
#                 + f"Action {i}: {action}\n"
#                 + f"Observation {i}: {obs}\n"
#             )

#             # Break, if done.
#             if done:
#                 break

#         return out


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
    def __init__(self, llm: Any, tools: Optional[List[BaseTool]] = [], prompt: Optional[str] = None) -> None:
        """Initialization."""
        super().__init__()
        self.llm = llm  # TODO: Why is `LLM` not usable here?
        self.tools = tools
        self.tools.append(search)
        prompt = hub.pull("hwchase17/react") if not prompt else prompt
        self.prompt = prompt
        if self.llm and self.tools and self.prompt:
            agent = create_react_agent(llm, tools, prompt)
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
