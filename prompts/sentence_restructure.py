from .base import SimpleAugmentation


class SentenceRestructure(SimpleAugmentation):
    @property
    def name(self) -> str:
        return "sentence_restructure"

    @property
    def prompt_template(self) -> str:
        return """Rewrite the text by restructuring sentences (passive→active, reordering clauses, combining/splitting sentences).

CRITICAL: Keep the exact same meaning and all information.
Do NOT add new details or change what is being described.

Input: {text}

Respond in JSON: {{"rewritten": "..."}}"""
