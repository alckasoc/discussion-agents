"""Base agent interface class."""

from abc import ABC
from typing import Any


class BaseAgent(ABC):
    """Base agent class providing a general interface for agent operations."""

    def reset(self, *args: Any, **kwargs: Any) -> Any:
        """Optionally resets the agent's state."""
        raise NotImplementedError("Reset method not implemented.")

    def plan(self, *args: Any, **kwargs: Any) -> Any:
        """Optionally implementable method to plan an action or set of actions."""
        raise NotImplementedError("Plan method not implemented.")

    def reflect(self, *args: Any, **kwargs: Any) -> Any:
        """Optionally implementable method to reflect on the current state or past actions."""
        raise NotImplementedError("Reflect method not implemented.")

    def score(self, *args: Any, **kwargs: Any) -> Any:
        """Optionally implementable method to score or evaluate something."""
        raise NotImplementedError("Score method not implemented.")

    def retrieve(self, *args: Any, **kwargs: Any) -> Any:
        """Optionally implementable method to retrieve memory."""
        raise NotImplementedError("Retrieve method not implemented.")

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        """Optionally implementable method to generate a response."""
        raise NotImplementedError("Generate method not implemented.")
