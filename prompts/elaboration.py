from .base import SimpleAugmentation


class Elaboration(SimpleAugmentation):
    @property
    def name(self) -> str:
        return "elaboration"

    @property
    def prompt_template(self) -> str:
        return """Expand the text by adding realistic details (filler words, elaborations, context).

CRITICAL: Do NOT add new problems or change alert types. Only add neutral context.
Make it about {target_pct}% longer.

Input: {text}

Respond in JSON: {{"rewritten": "..."}}"""
