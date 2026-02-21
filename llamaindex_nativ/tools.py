"""LlamaIndex tool spec for the Nativ AI localization platform.

Provides a ``NativToolSpec`` that wraps the ``nativ`` Python SDK and
exposes every method as a LlamaIndex ``FunctionTool``.
"""

from __future__ import annotations

from typing import List, Optional

from llama_index.core.tools.tool_spec.base import BaseToolSpec

import nativ as _nativ_sdk


def _fmt_translation(t: _nativ_sdk.Translation) -> str:
    parts = [t.translated_text]
    if t.rationale:
        parts.append(f"Rationale: {t.rationale}")
    if t.backtranslation:
        parts.append(f"Back-translation: {t.backtranslation}")
    if t.tm_match and t.tm_match.score > 0:
        parts.append(
            f"TM match: {t.tm_match.score:.0f}% ({t.tm_match.match_type})"
        )
    return "\n".join(parts)


class NativToolSpec(BaseToolSpec):
    """LlamaIndex tool spec for Nativ -- AI-powered localization.

    Usage::

        from llamaindex_nativ import NativToolSpec

        tools = NativToolSpec().to_tool_list()       # reads NATIV_API_KEY from env
        tools = NativToolSpec(api_key="nativ_...").to_tool_list()
    """

    spec_functions = [
        "translate",
        "translate_batch",
        "search_translation_memory",
        "add_translation_memory_entry",
        "get_languages",
        "get_style_guides",
        "get_brand_voice",
        "get_translation_memory_stats",
    ]

    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        base_url: Optional[str] = None,
    ) -> None:
        super().__init__()
        self._api_key = api_key
        self._base_url = base_url

    def _client(self) -> _nativ_sdk.Nativ:
        return _nativ_sdk.Nativ(api_key=self._api_key, base_url=self._base_url)

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------

    def translate(
        self,
        text: str,
        target_language: str,
        target_language_code: Optional[str] = None,
        source_language: str = "English",
        source_language_code: str = "en",
        context: Optional[str] = None,
        glossary: Optional[str] = None,
        formality: Optional[str] = None,
        max_characters: Optional[int] = None,
        backtranslate: bool = False,
    ) -> str:
        """Translate text using Nativ's AI localization engine.

        Automatically leverages the team's translation memory, brand voice,
        and style guides for consistent, on-brand translations.

        Args:
            text: The text to translate.
            target_language: Full target language name, e.g. 'French', 'German'.
            target_language_code: ISO language code, e.g. 'fr'. Auto-detected if omitted.
            source_language: Source language name.
            source_language_code: Source language ISO code.
            context: Context to guide the translation, e.g. 'mobile app button'.
            glossary: Inline glossary as CSV, e.g. 'term,translation\\nbrand,marque'.
            formality: Tone: very_informal | informal | neutral | formal | very_formal.
            max_characters: Strict character limit for the output.
            backtranslate: If true, also return a back-translation to verify intent.
        """
        with self._client() as c:
            result = c.translate(
                text,
                target_language,
                target_language_code=target_language_code,
                source_language=source_language,
                source_language_code=source_language_code,
                context=context,
                glossary=glossary,
                formality=formality,
                max_characters=max_characters,
                backtranslate=backtranslate,
            )
        return _fmt_translation(result)

    def translate_batch(
        self,
        texts: List[str],
        target_language: str,
        target_language_code: Optional[str] = None,
        source_language: str = "English",
        source_language_code: str = "en",
        context: Optional[str] = None,
        formality: Optional[str] = None,
    ) -> str:
        """Translate multiple texts to the same target language in one call.

        Args:
            texts: List of texts to translate.
            target_language: Full target language name, e.g. 'French'.
            target_language_code: ISO language code.
            source_language: Source language name.
            source_language_code: Source language code.
            context: Context hint for all translations.
            formality: Tone: very_informal | informal | neutral | formal | very_formal.
        """
        with self._client() as c:
            results = c.translate_batch(
                texts,
                target_language,
                target_language_code=target_language_code,
                source_language=source_language,
                source_language_code=source_language_code,
                context=context,
                formality=formality,
            )
        return "\n".join(
            f"{i + 1}. {r.translated_text}" for i, r in enumerate(results)
        )

    def search_translation_memory(
        self,
        query: str,
        source_language_code: str = "en",
        target_language_code: Optional[str] = None,
        min_score: float = 0.0,
        limit: int = 10,
    ) -> str:
        """Fuzzy-search existing translations in the translation memory.

        Args:
            query: Text to search for.
            source_language_code: Source language code.
            target_language_code: Target language code to filter results.
            min_score: Minimum fuzzy-match score (0-100).
            limit: Maximum number of results.
        """
        with self._client() as c:
            matches = c.search_tm(
                query,
                source_language_code=source_language_code,
                target_language_code=target_language_code,
                min_score=min_score,
                limit=limit,
            )
        if not matches:
            return "No matches found in translation memory."
        lines = [f"Found {len(matches)} match(es):"]
        for m in matches:
            lines.append(
                f'- [{m.score:.0f}% {m.match_type}] '
                f'"{m.source_text}" -> "{m.target_text}"'
            )
        return "\n".join(lines)

    def add_translation_memory_entry(
        self,
        source_text: str,
        target_text: str,
        source_language_code: str,
        target_language_code: str,
        name: Optional[str] = None,
    ) -> str:
        """Store an approved translation in the translation memory for reuse.

        Args:
            source_text: The original text.
            target_text: The approved translation.
            source_language_code: Source language code, e.g. 'en'.
            target_language_code: Target language code, e.g. 'fr'.
            name: Optional label for this entry, e.g. 'homepage hero copy'.
        """
        with self._client() as c:
            entry = c.add_tm_entry(
                source_text,
                target_text,
                source_language_code,
                target_language_code,
                name=name,
            )
        return (
            f"Added TM entry {entry.id}: "
            f'"{entry.source_text}" ({entry.source_language_code}) -> '
            f'"{entry.target_text}" ({entry.target_language_code})'
        )

    def get_languages(self) -> str:
        """List all target languages configured in the Nativ workspace."""
        with self._client() as c:
            langs = c.get_languages()
        if not langs:
            return "No languages configured."
        lines = ["Configured languages:"]
        for lang in langs:
            entry = f"- {lang.language} ({lang.language_code})"
            if lang.formality:
                entry += f" -- formality: {lang.formality}"
            lines.append(entry)
        return "\n".join(lines)

    def get_style_guides(self) -> str:
        """Get all style guides configured in the workspace."""
        with self._client() as c:
            guides = c.get_style_guides()
        if not guides:
            return "No style guides configured."
        lines = [f"Style guides ({len(guides)}):"]
        for g in guides:
            status = "enabled" if g.is_enabled else "disabled"
            lines.append(f"\n## {g.title} [{status}]\n{g.content}")
        return "\n".join(lines)

    def get_brand_voice(self) -> str:
        """Get the brand voice prompt that shapes all translations."""
        with self._client() as c:
            bv = c.get_brand_voice()
        if not bv.exists or not bv.prompt:
            return "No brand voice configured."
        return f"Brand voice:\n{bv.prompt}"

    def get_translation_memory_stats(self) -> str:
        """Get translation memory statistics: total entries, enabled/disabled, breakdown by source."""
        with self._client() as c:
            stats = c.get_tm_stats()
        lines = [
            f"Translation memory: {stats.total} total entries",
            f"  Enabled: {stats.enabled}",
            f"  Disabled: {stats.disabled}",
        ]
        if stats.by_source:
            lines.append("  By source:")
            for source, counts in stats.by_source.items():
                lines.append(f"    {source}: {counts}")
        return "\n".join(lines)
