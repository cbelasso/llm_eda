from .base import BaseAugmentation, ReferenceBasedAugmentation, SimpleAugmentation
from .compression import Compression
from .elaboration import Elaboration
from .formality_shift import FormalityShift
from .noise_injection import NoiseInjection
from .paraphrase import Paraphrase
from .sentence_restructure import SentenceRestructure
from .style_rewrite import StyleRewrite
from .synonym_replace import SynonymReplace

# Registry of all available augmentations
AUGMENTATION_REGISTRY = {
    "style_rewrite": StyleRewrite(),
    "synonym_replace": SynonymReplace(),
    "sentence_restructure": SentenceRestructure(),
    "noise_injection": NoiseInjection(),
    "paraphrase": Paraphrase(),
    "formality_shift": FormalityShift(),
    "elaboration": Elaboration(),
    "compression": Compression(),
}


def get_augmentation(name: str) -> BaseAugmentation:
    """Get an augmentation by name."""
    if name not in AUGMENTATION_REGISTRY:
        raise ValueError(
            f"Unknown augmentation: {name}. Available: {list(AUGMENTATION_REGISTRY.keys())}"
        )
    return AUGMENTATION_REGISTRY[name]


__all__ = [
    "BaseAugmentation",
    "ReferenceBasedAugmentation",
    "SimpleAugmentation",
    "StyleRewrite",
    "SynonymReplace",
    "SentenceRestructure",
    "NoiseInjection",
    "Paraphrase",
    "FormalityShift",
    "Elaboration",
    "Compression",
    "AUGMENTATION_REGISTRY",
    "get_augmentation",
]
