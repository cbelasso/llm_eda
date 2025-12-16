from .base import SimpleAugmentation


class SynonymReplace(SimpleAugmentation):
    @property
    def name(self) -> str:
        return "synonym_replace"

    @property
    def prompt_template(self) -> str:
        return """Rewrite the text by replacing 3-5 words with contextually appropriate synonyms.

CRITICAL: Preserve exact meaning, tone, and severity of any issues mentioned.
Do NOT add, remove, or change any problems described.

Input: {text}

Respond in JSON: {{"rewritten": "..."}}"""
