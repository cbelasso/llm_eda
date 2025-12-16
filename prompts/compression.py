from .base import SimpleAugmentation


class Compression(SimpleAugmentation):
    @property
    def name(self) -> str:
        return "compression"

    @property
    def prompt_template(self) -> str:
        return """Compress the text to be more concise while keeping all key information.

CRITICAL: Do NOT remove any problematic content or change severity.
Make it about {target_pct}% shorter.

Input: {text}

Respond in JSON: {{"rewritten": "..."}}"""
