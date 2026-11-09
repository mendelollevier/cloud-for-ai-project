"""Gedeelde fixtures: één keer data inlezen en één keer trainen, voor alle tests."""

import pytest

from src.config import laad_config
from src.data import laad_seizoenen
from src.train import laad_en_maak_features, train


@pytest.fixture(scope="session")
def cfg():
    return laad_config()


@pytest.fixture(scope="session")
def ruw(cfg):
    """De seizoenen zoals ze ingelezen worden, nog zonder features."""
    return laad_seizoenen(cfg["data"]["pad"])


@pytest.fixture(scope="session")
def df(cfg):
    """Gevalideerd en met features: wat het model te zien krijgt."""
    return laad_en_maak_features(cfg)


@pytest.fixture(scope="session")
def model_en_scores(df, cfg):
    return train(df, cfg)


@pytest.fixture(scope="session")
def model(model_en_scores):
    return model_en_scores[0]
