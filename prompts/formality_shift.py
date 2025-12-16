from .base import SimpleAugmentation


class FormalityShift(SimpleAugmentation):
    @property
    def name(self) -> str:
        return "formality_shift"

    @property
    def prompt_template(self) -> str:
        return """Rewrite the text to be {direction}.

CRITICAL: Only change tone and style, not content or severity of issues.

Input: {text}

Respond in JSON: {{"rewritten": "..."}}"""
