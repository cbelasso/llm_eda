from .base import SimpleAugmentation


class Paraphrase(SimpleAugmentation):
    @property
    def name(self) -> str:
        return "paraphrase"

    @property
    def prompt_template(self) -> str:
        return """Paraphrase the text using different words and phrasing.

CRITICAL: Preserve exact semantic content. If it mentions discrimination, harassment,
or specific issues, those MUST remain unchanged in severity and type.

Input: {text}

Respond in JSON: {{"rewritten": "..."}}"""
