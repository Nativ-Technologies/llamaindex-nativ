"""Unit tests for llamaindex-nativ tools.

All tests mock the underlying nativ SDK client so no API key or network
access is needed.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from llamaindex_nativ import NativToolSpec

FAKE_KEY = "nativ_test_00000000000000000000000000000000"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _translation(**overrides):
    from nativ import Translation, TranslationMetadata

    defaults = dict(
        translated_text="Bonjour le monde",
        metadata=TranslationMetadata(word_count=2, cost=1),
        tm_match=None,
        rationale="Standard greeting.",
        backtranslation=None,
    )
    defaults.update(overrides)
    return Translation(**defaults)


def _tm_search_match(**overrides):
    from nativ import TMSearchMatch

    defaults = dict(
        tm_id="tm_1",
        score=95.0,
        match_type="fuzzy",
        source_text="Hello",
        target_text="Bonjour",
        information_source="manual",
    )
    defaults.update(overrides)
    return TMSearchMatch(**defaults)


def _tm_entry(**overrides):
    from nativ import TMEntry

    defaults = dict(
        id="entry_1",
        source_language_code="en",
        source_text="Hello",
        target_language_code="fr",
        target_text="Bonjour",
        information_source="manual",
        enabled=True,
        priority=50,
    )
    defaults.update(overrides)
    return TMEntry(**defaults)


def _language(**overrides):
    from nativ import Language

    defaults = dict(id=1, language="French", language_code="fr", formality="formal")
    defaults.update(overrides)
    return Language(**defaults)


def _style_guide(**overrides):
    from nativ import StyleGuide

    defaults = dict(
        id="sg_1",
        title="Tone",
        content="Use a warm, friendly tone.",
        is_enabled=True,
    )
    defaults.update(overrides)
    return StyleGuide(**defaults)


def _brand_voice(**overrides):
    from nativ import BrandVoice

    defaults = dict(prompt="Be concise and friendly.", exists=True)
    defaults.update(overrides)
    return BrandVoice(**defaults)


def _tm_stats(**overrides):
    from nativ import TMStats

    defaults = dict(total=100, enabled=90, disabled=10, by_source={})
    defaults.update(overrides)
    return TMStats(**defaults)


# ---------------------------------------------------------------------------
# NativToolSpec basics
# ---------------------------------------------------------------------------


class TestNativToolSpec:
    def test_to_tool_list_returns_eight(self):
        spec = NativToolSpec(api_key=FAKE_KEY)
        tools = spec.to_tool_list()
        assert len(tools) == 8

    def test_tool_names(self):
        spec = NativToolSpec(api_key=FAKE_KEY)
        tools = spec.to_tool_list()
        names = {t.metadata.name for t in tools}
        assert "translate" in names
        assert "translate_batch" in names
        assert "search_translation_memory" in names
        assert "add_translation_memory_entry" in names
        assert "get_languages" in names
        assert "get_style_guides" in names
        assert "get_brand_voice" in names
        assert "get_translation_memory_stats" in names

    def test_spec_functions_count(self):
        assert len(NativToolSpec.spec_functions) == 8


# ---------------------------------------------------------------------------
# translate
# ---------------------------------------------------------------------------


class TestTranslate:
    def test_basic_translate(self):
        spec = NativToolSpec(api_key=FAKE_KEY)
        mock_client = MagicMock()
        mock_client.translate.return_value = _translation()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        with patch.object(spec, "_client", return_value=mock_client):
            result = spec.translate("Hello world", target_language="French")

        assert "Bonjour le monde" in result
        assert "Rationale: Standard greeting." in result
        mock_client.translate.assert_called_once()

    def test_translate_with_backtranslation(self):
        spec = NativToolSpec(api_key=FAKE_KEY)
        mock_client = MagicMock()
        mock_client.translate.return_value = _translation(
            backtranslation="Hello world"
        )
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        with patch.object(spec, "_client", return_value=mock_client):
            result = spec.translate(
                "Hello world", target_language="French", backtranslate=True
            )

        assert "Back-translation: Hello world" in result


# ---------------------------------------------------------------------------
# translate_batch
# ---------------------------------------------------------------------------


class TestTranslateBatch:
    def test_batch(self):
        spec = NativToolSpec(api_key=FAKE_KEY)
        mock_client = MagicMock()
        mock_client.translate_batch.return_value = [
            _translation(translated_text="Bonjour"),
            _translation(translated_text="Au revoir"),
        ]
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        with patch.object(spec, "_client", return_value=mock_client):
            result = spec.translate_batch(
                ["Hello", "Goodbye"], target_language="French"
            )

        assert "1. Bonjour" in result
        assert "2. Au revoir" in result


# ---------------------------------------------------------------------------
# search_translation_memory
# ---------------------------------------------------------------------------


class TestSearchTM:
    def test_with_matches(self):
        spec = NativToolSpec(api_key=FAKE_KEY)
        mock_client = MagicMock()
        mock_client.search_tm.return_value = [_tm_search_match()]
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        with patch.object(spec, "_client", return_value=mock_client):
            result = spec.search_translation_memory("Hello")

        assert "1 match" in result
        assert "Hello" in result
        assert "Bonjour" in result

    def test_no_matches(self):
        spec = NativToolSpec(api_key=FAKE_KEY)
        mock_client = MagicMock()
        mock_client.search_tm.return_value = []
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        with patch.object(spec, "_client", return_value=mock_client):
            result = spec.search_translation_memory("xyzzy")

        assert "No matches found" in result


# ---------------------------------------------------------------------------
# add_translation_memory_entry
# ---------------------------------------------------------------------------


class TestAddTMEntry:
    def test_add_entry(self):
        spec = NativToolSpec(api_key=FAKE_KEY)
        mock_client = MagicMock()
        mock_client.add_tm_entry.return_value = _tm_entry()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        with patch.object(spec, "_client", return_value=mock_client):
            result = spec.add_translation_memory_entry(
                "Hello", "Bonjour", "en", "fr"
            )

        assert "Added TM entry" in result
        assert "Hello" in result
        assert "Bonjour" in result


# ---------------------------------------------------------------------------
# get_languages
# ---------------------------------------------------------------------------


class TestGetLanguages:
    def test_with_languages(self):
        spec = NativToolSpec(api_key=FAKE_KEY)
        mock_client = MagicMock()
        mock_client.get_languages.return_value = [
            _language(),
            _language(id=2, language="German", language_code="de", formality=None),
        ]
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        with patch.object(spec, "_client", return_value=mock_client):
            result = spec.get_languages()

        assert "French (fr)" in result
        assert "formality: formal" in result
        assert "German (de)" in result

    def test_empty(self):
        spec = NativToolSpec(api_key=FAKE_KEY)
        mock_client = MagicMock()
        mock_client.get_languages.return_value = []
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        with patch.object(spec, "_client", return_value=mock_client):
            result = spec.get_languages()

        assert "No languages configured" in result


# ---------------------------------------------------------------------------
# get_style_guides
# ---------------------------------------------------------------------------


class TestGetStyleGuides:
    def test_with_guides(self):
        spec = NativToolSpec(api_key=FAKE_KEY)
        mock_client = MagicMock()
        mock_client.get_style_guides.return_value = [_style_guide()]
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        with patch.object(spec, "_client", return_value=mock_client):
            result = spec.get_style_guides()

        assert "Tone" in result
        assert "warm, friendly" in result

    def test_empty(self):
        spec = NativToolSpec(api_key=FAKE_KEY)
        mock_client = MagicMock()
        mock_client.get_style_guides.return_value = []
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        with patch.object(spec, "_client", return_value=mock_client):
            result = spec.get_style_guides()

        assert "No style guides configured" in result


# ---------------------------------------------------------------------------
# get_brand_voice
# ---------------------------------------------------------------------------


class TestGetBrandVoice:
    def test_with_voice(self):
        spec = NativToolSpec(api_key=FAKE_KEY)
        mock_client = MagicMock()
        mock_client.get_brand_voice.return_value = _brand_voice()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        with patch.object(spec, "_client", return_value=mock_client):
            result = spec.get_brand_voice()

        assert "concise and friendly" in result

    def test_no_voice(self):
        spec = NativToolSpec(api_key=FAKE_KEY)
        mock_client = MagicMock()
        mock_client.get_brand_voice.return_value = _brand_voice(
            exists=False, prompt=None
        )
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        with patch.object(spec, "_client", return_value=mock_client):
            result = spec.get_brand_voice()

        assert "No brand voice configured" in result


# ---------------------------------------------------------------------------
# get_translation_memory_stats
# ---------------------------------------------------------------------------


class TestGetTMStats:
    def test_stats(self):
        spec = NativToolSpec(api_key=FAKE_KEY)
        mock_client = MagicMock()
        mock_client.get_tm_stats.return_value = _tm_stats()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        with patch.object(spec, "_client", return_value=mock_client):
            result = spec.get_translation_memory_stats()

        assert "100 total entries" in result
        assert "Enabled: 90" in result
        assert "Disabled: 10" in result
