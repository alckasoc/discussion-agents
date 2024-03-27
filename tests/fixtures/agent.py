"""Fixtures for creating agents."""

import pytest

from langchain_community.chat_models.fake import FakeListChatModel
from langchain_community.llms.fake import FakeListLLM

from discussion_agents.cog.agent.generative_agents import GenerativeAgent
from discussion_agents.cog.agent.react import ReActAgent
from discussion_agents.cog.agent.reflexion import ReflexionCoTAgent, ReflexionReActAgent
from discussion_agents.cog.agent.expel import ExpeLAgent


@pytest.fixture
def generative_agent() -> GenerativeAgent:
    """Creates a GenerativeAgent."""
    agent = GenerativeAgent(llm=FakeListLLM(responses=["1"]))
    return agent


@pytest.fixture
def react_agent() -> ReActAgent:
    """Creates a ReActAgent."""
    agent = ReActAgent(llm=FakeListLLM(responses=["1"]))
    return agent


@pytest.fixture
def reflexion_cot_agent() -> ReflexionCoTAgent:
    """Creates a ReflexionCoTAgent."""
    agent = ReflexionCoTAgent(
        self_reflect_llm=FakeListChatModel(responses=["1"]),
        action_llm=FakeListChatModel(responses=["1"]),
    )
    return agent


@pytest.fixture
def reflexion_react_agent() -> ReflexionReActAgent:
    """Creates a ReflexionReActAgent."""
    agent = ReflexionReActAgent(
        self_reflect_llm=FakeListChatModel(responses=["1"]),
        action_llm=FakeListChatModel(responses=["1"]),
        max_trials=1,
        max_reflections=3,
    )
    return agent


@pytest.fixture
def expel_agent() -> ExpeLAgent:
    """Creates a ExpeLAgent."""
    agent = ExpeLAgent(
        llm=FakeListChatModel(responses=["1"])
    )
    return agent