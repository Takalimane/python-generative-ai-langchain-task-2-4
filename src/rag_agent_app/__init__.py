"""Grounded RAG and tool agent application."""


def safe_external_text(value: str) -> str:
    """Normalize smart Unicode punctuation before ASCII-sensitive integrations."""
    replacements = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2026": "...",
        "\u00a0": " ",
    }
    result = value
    for source, target in replacements.items():
        result = result.replace(source, target)
    return result.encode("utf-8", "replace").decode("utf-8")
