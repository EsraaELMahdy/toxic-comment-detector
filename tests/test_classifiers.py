from __future__ import annotations

import pytest

import config
from classifiers.base import ModelStatus
from classifiers.registry import all_classifiers, get_classifier, model_choices

CHECKPOINT_PRESENT = config.resolve_lstm_checkpoint() is not None
requires_checkpoint = pytest.mark.skipif(
    not CHECKPOINT_PRESENT,
    reason="LSTM checkpoint not present in models/lstm/",
)


class TestRegistry:
    def test_registers_two_models(self):
        assert len(all_classifiers()) == 2

    def test_model_ids_are_unique(self):
        ids = [classifier.model_id for classifier in all_classifiers()]
        assert len(ids) == len(set(ids))

    def test_expected_models_are_registered(self):
        assert {model_id for model_id, _ in model_choices()} == {"lstm", "albert_lora"}

    def test_get_classifier_returns_singletons(self):
        assert get_classifier("lstm") is get_classifier("lstm")

    def test_unknown_model_raises(self):
        with pytest.raises(KeyError):
            get_classifier("does-not-exist")


class TestAvailabilityContract:
    @pytest.mark.parametrize("classifier", all_classifiers(), ids=lambda c: c.model_id)
    def test_every_model_reports_a_status_and_a_reason(self, classifier):
        status, reason = classifier.availability()
        assert isinstance(status, ModelStatus)
        assert isinstance(reason, str) and reason

    def test_albert_reports_missing_weights_for_empty_dir(self, tmp_path):
        from classifiers.albert_classifier import AlbertLoraClassifier

        status, _ = AlbertLoraClassifier(tmp_path / "not_trained").availability()
        assert status is ModelStatus.MISSING_WEIGHTS

    def test_albert_ignores_placeholder_files(self, tmp_path):
        from classifiers.albert_classifier import AlbertLoraClassifier

        folder = tmp_path / "albert_lora"
        folder.mkdir()
        (folder / ".gitkeep").write_text("")

        status, _ = AlbertLoraClassifier(folder).availability()
        assert status is ModelStatus.MISSING_WEIGHTS

    def test_albert_reports_missing_packages(self, tmp_path, monkeypatch):
        from classifiers.albert_classifier import AlbertLoraClassifier

        folder = tmp_path / "albert_lora"
        folder.mkdir()
        (folder / "adapter_config.json").write_text("{}")

        classifier = AlbertLoraClassifier(folder)
        monkeypatch.setattr(classifier, "_missing_packages", lambda: ["sentencepiece"])

        status, reason = classifier.availability()
        assert status is ModelStatus.UNAVAILABLE
        assert "sentencepiece" in reason
        assert "pip install" in reason


@requires_checkpoint
class TestLSTMInference:
    @pytest.fixture(scope="class")
    def classifier(self):
        model = get_classifier("lstm")
        model.load()
        return model

    def test_is_ready(self):
        assert get_classifier("lstm").availability()[0] is ModelStatus.READY

    def test_clean_comment_is_not_flagged(self, classifier):
        result = classifier.predict(
            "Thank you so much for the detailed explanation, that really helped me."
        )
        assert result.is_clean, f"expected clean, got {result.verdict}"

    def test_insult_is_flagged(self, classifier):
        result = classifier.predict(
            "You are a complete idiot and nobody here wants your useless opinion."
        )
        assert not result.is_clean

    def test_result_shape_matches_the_contract(self, classifier):
        result = classifier.predict("hello there")

        assert result.model_id == "lstm"
        assert result.model_label == "LSTM"
        assert len(result.predictions) == len(config.LABELS)
        assert result.latency_ms > 0
        for prediction in result.predictions:
            assert 0.0 <= prediction.probability <= 1.0
            assert 0.0 < prediction.threshold < 1.0

    def test_is_deterministic(self, classifier):
        text = "This is a fairly ordinary sentence about the weather."
        first = classifier.predict(text)
        second = classifier.predict(text)
        assert first.verdict == second.verdict
        assert [p.probability for p in first.predictions] == [
            p.probability for p in second.predictions
        ]

    def test_handles_empty_input(self, classifier):
        assert classifier.predict("") is not None

    def test_handles_very_long_input(self, classifier):
        assert classifier.predict("word " * 2000) is not None

    def test_handles_unicode_input(self, classifier):
        assert classifier.predict("مرحبا بالعالم 🎉 café") is not None


class TestSentencepieceDetection:
    def _folder(self, tmp_path, files):
        folder = tmp_path / "albert_lora"
        folder.mkdir()
        (folder / "adapter_config.json").write_text("{}")
        for name in files:
            (folder / name).write_text("")
        return folder

    def test_fast_tokenizer_needs_no_sentencepiece(self, tmp_path):
        from classifiers.albert_classifier import AlbertLoraClassifier

        folder = self._folder(tmp_path, ["tokenizer.json", "tokenizer_config.json"])
        assert AlbertLoraClassifier(folder)._needs_sentencepiece() is False

    def test_spiece_model_requires_sentencepiece(self, tmp_path):
        from classifiers.albert_classifier import AlbertLoraClassifier

        folder = self._folder(tmp_path, ["spiece.model", "tokenizer_config.json"])
        assert AlbertLoraClassifier(folder)._needs_sentencepiece() is True

    def test_unknown_tokenizer_defaults_to_requiring_sentencepiece(self, tmp_path):
        from classifiers.albert_classifier import AlbertLoraClassifier

        folder = self._folder(tmp_path, ["tokenizer_config.json"])
        assert AlbertLoraClassifier(folder)._needs_sentencepiece() is True
