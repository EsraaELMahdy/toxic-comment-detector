from __future__ import annotations

from collections.abc import Callable

from classifiers.albert_classifier import AlbertLoraClassifier
from classifiers.base import BaseClassifier, ModelStatus
from classifiers.lstm_classifier import LSTMClassifier

_FACTORIES: dict[str, Callable[[], BaseClassifier]] = {
    LSTMClassifier.model_id: LSTMClassifier,
    AlbertLoraClassifier.model_id: AlbertLoraClassifier,
}

_instances: dict[str, BaseClassifier] = {}


def get_classifier(model_id: str) -> BaseClassifier:
    if model_id not in _FACTORIES:
        raise KeyError(f"Unknown classifier: {model_id!r}")

    if model_id not in _instances:
        _instances[model_id] = _FACTORIES[model_id]()

    return _instances[model_id]


def all_classifiers() -> list[BaseClassifier]:
    return [get_classifier(model_id) for model_id in _FACTORIES]


def ready_classifiers() -> list[BaseClassifier]:
    return [
        classifier
        for classifier in all_classifiers()
        if classifier.availability()[0] is ModelStatus.READY
    ]


def model_choices() -> list[tuple[str, str]]:
    return [(classifier.model_id, classifier.display_name) for classifier in all_classifiers()]
