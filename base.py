# models/base.py
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

class ModelInterface(ABC):
    """
    Abstract interface that every model adapter must implement.
    This enforces: initialization, prompt template, and a `respond` method.
    """

    @abstractmethod
    def initialize(self) -> None:
        """Initialize any internal model clients / connections."""
        raise NotImplementedError

    @abstractmethod
    def get_prompt_template(self) -> Any:
        """
        Return a prompt template object (e.g., LangChain PromptTemplate).
        The template must accept system_prompt, user_prompt and a format instructions placeholder.
        """
        raise NotImplementedError

    @abstractmethod
    def respond(
        self,
        system_prompt: str,
        user_prompt: str,
        format_instructions: str,
        multimodal_inputs: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Execute the model call and return the parsed output. `multimodal_inputs` is a dict
        that may contain keys like 'image_url' or 'image_bytes' for future extension.
        Return value may be:
          - parsed Pydantic object (if JSON parser used)
          - raw string (if parsing fails or user chose RAW output)
        """
        raise NotImplementedError
