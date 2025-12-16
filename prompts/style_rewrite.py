from .base import ReferenceBasedAugmentation


class StyleRewrite(ReferenceBasedAugmentation):
    @property
    def name(self) -> str:
        return "style_rewrite"

    @property
    def prompt_template(self) -> str:
        return """You are given two texts:

1. A reference text that defines the target **stylistic attributes** (tone, structure, phrasing, vocabulary, elaboration, grammatical tendencies, punctuation tendencies, whitespaces, indentation, extraneous punctuation, abbreviations, emojis, shoutouts, capitalization, misspelling, redundancies, etc.)
2. An input comment that defines the **content** to preserve.

Your task is to rewrite the input comment to mimic the **style** of the reference text **without adding, inferring, or expanding** any meaning beyond what is explicitly in the input. 
If the input is very short, keep it short — only modify surface features (e.g., capitalization, formality, punctuation) to align with the reference style. 
Never insert new ideas, explanations, actors, or any other superfluous information.

Reference Text:
{reference}

Input Text:
{text}

CRITICAL: Preserve exact semantic content and severity of any issues mentioned.

Respond only in JSON format:
{{"rewritten": "<your rewritten version>"}}"""
