from __future__ import annotations

import json

from core.schemas import ClassificationResult, LabelPrediction


def build(*triples) -> ClassificationResult:
    predictions = [
        LabelPrediction(label=label, probability=probability, threshold=threshold)
        for label, probability, threshold in triples
    ]
    return ClassificationResult(
        model_id="test",
        model_label="Test",
        input_text="sample",
        predictions=predictions,
    )


class TestLabelPrediction:
    def test_positive_when_above_threshold(self):
        assert LabelPrediction("toxic", 0.9, 0.5).positive is True

    def test_positive_when_exactly_on_threshold(self):
        assert LabelPrediction("toxic", 0.5, 0.5).positive is True

    def test_negative_when_below_threshold(self):
        assert LabelPrediction("toxic", 0.49, 0.5).positive is False

    def test_display_name_is_friendly(self):
        assert LabelPrediction("obscene", 0.1, 0.5).display_name == "Obscene"


class TestClassificationResult:
    def test_clean_when_nothing_crosses(self):
        result = build(("toxic", 0.1, 0.5), ("insult", 0.2, 0.5))
        assert result.is_clean is True
        assert result.verdict == "non-toxic"
        assert result.severity == 0.0

    def test_reports_every_detected_label(self):
        result = build(("toxic", 0.9, 0.5), ("obscene", 0.8, 0.5), ("insult", 0.1, 0.5))
        assert result.verdict == "toxic, obscene"
        assert len(result.detected) == 2

    def test_severity_is_the_peak_probability(self):
        result = build(("toxic", 0.67, 0.5), ("insult", 0.93, 0.5))
        assert result.severity == 0.93

    def test_custom_thresholds_are_respected(self):
        result = build(("toxic", 0.6, 0.7))
        assert result.is_clean is True

    def test_probabilities_json_is_valid(self):
        payload = json.loads(build(("toxic", 0.91234, 0.5)).probabilities_json)
        assert payload == {"toxic": 0.9123}
