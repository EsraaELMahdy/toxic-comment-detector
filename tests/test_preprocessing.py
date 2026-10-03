from __future__ import annotations

import pytest

from core.preprocessing import clean_text, encode_for_lstm, tokenize


class TestCleanText:
    def test_lowercases(self):
        assert clean_text("HELLO World") == "hello world"

    def test_strips_html(self):
        assert "<b>" not in clean_text("this is <b>bold</b> text")

    def test_removes_urls(self):
        cleaned = clean_text("visit https://example.com/page now")
        assert "http" not in cleaned
        assert cleaned == "visit now"

    def test_removes_non_letters(self):
        assert clean_text("wow!!! 123 $$$") == "wow"

    def test_collapses_whitespace(self):
        assert clean_text("too    many      spaces") == "too many spaces"

    def test_handles_non_string_input(self):
        assert clean_text(None) == "none"

    def test_empty_input(self):
        assert clean_text("") == ""


class TestTokenize:
    def test_splits_and_lemmatises(self):
        assert tokenize("The dogs are running") == ["the", "dog", "are", "running"]

    def test_is_deterministic(self):
        text = "these sentences are definitely being watched"
        assert tokenize(text) == tokenize(text)

    def test_empty_input(self):
        assert tokenize("") == []


class TestEncodeForLstm:
    VOCAB = {"<UNK>": 1, "hello": 7, "world": 9}

    def test_maps_known_tokens(self):
        assert encode_for_lstm("hello world", self.VOCAB, max_len=5)[:2] == [7, 9]

    def test_unknown_tokens_fall_back(self):
        assert encode_for_lstm("unseen", self.VOCAB, max_len=3)[0] == 1

    def test_pads_to_max_len(self):
        assert len(encode_for_lstm("hello", self.VOCAB, max_len=8)) == 8

    def test_truncates_long_input(self):
        assert len(encode_for_lstm("hello world hello world", self.VOCAB, max_len=3)) == 3

    @pytest.mark.parametrize("max_len", [1, 16, 64, 200])
    def test_always_returns_exact_length(self, max_len):
        assert len(encode_for_lstm("a b c", self.VOCAB, max_len=max_len)) == max_len
