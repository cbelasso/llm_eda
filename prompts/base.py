from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BaseAugmentation(ABC):
    """Base class for all augmentation types."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name for this augmentation type."""
        pass

    @property
    @abstractmethod
    def requires_reference(self) -> bool:
        """Whether this augmentation needs a reference text."""
        pass

    @abstractmethod
    def build_prompt(self, text: str, params: Optional[Dict[str, Any]] = None) -> str:
        """Build the prompt for this augmentation."""
        pass

    @property
    @abstractmethod
    def prompt_template(self) -> str:
        """The prompt template string."""
        pass


class ReferenceBasedAugmentation(BaseAugmentation):
    """Base class for augmentations that need a reference text."""

    @property
    def requires_reference(self) -> bool:
        return True

    def build_prompt(self, text: str, params: Optional[Dict[str, Any]] = None) -> str:
        params = params or {}
        if "reference" not in params:
            raise ValueError(f"{self.name} requires 'reference' in params")
        return self.prompt_template.format(text=text, **params)


class SimpleAugmentation(BaseAugmentation):
    """Base class for augmentations that don't need a reference."""

    @property
    def requires_reference(self) -> bool:
        return False

    def build_prompt(self, text: str, params: Optional[Dict[str, Any]] = None) -> str:
        params = params or {}
        return self.prompt_template.format(text=text, **params)
