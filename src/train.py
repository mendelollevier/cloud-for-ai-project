"""Drie modellen trainen, vergelijken op log loss en het beste wegschrijven.

python -m src.train
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import schema
from .config import alle_features, laad_config
from .data import laad_seizoenen
from .features import maak_features


def bouw_pijplijn(cfg: dict) -> Pipeline:
    """Alle stappen die iets leren uit de data zitten ín de pijplijn. Zie labo 4."""
    voorbewerking = ColumnTransformer(
        [
            # Eerst lege vorm invullen met de mediaan (alleen geleerd uit train), dan schalen.
            (
                "num",
                Pipeline(
                    [("invullen", SimpleImputer(strategy="median")), ("schalen", StandardScaler())]
                ),
                cfg["features"]["numeriek"],
            ),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cfg["features"]["categorisch"]),
        ]
    )
    model = LogisticRegression(max_iter=cfg["model"]["max_iter"])
    return Pipeline([("voorbewerking", voorbewerking), ("model", model)])


def laad_en_maak_features(cfg: dict) -> pd.DataFrame:
    """Ruwe seizoenen inlezen, valideren met het schema en de features toevoegen."""
    ruw = laad_seizoenen(cfg["data"]["pad"])
    schema.GESPEELD.validate(ruw, lazy=True)
    return maak_features(ruw)


def splits(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split per seizoen, in de tijd: train is altijd ouder dan validatie, validatie ouder dan test.

    Nooit willekeurig door elkaar (les 4, slide 32-33): anders leert het model uit de toekomst.
    """
    s = cfg["split"]
    train = df[df["seizoen"].isin(s["train"])]
    validatie = df[df["seizoen"] == s["validatie"]]
    test = df[df["seizoen"] == s["test"]]
    return train, validatie, test


def kans_bet365(df: pd.DataFrame) -> pd.Series:
    """Baseline 2: de kans op thuiswinst volgens Bet365, zonder de marge van de bookmaker.

    1/odds telt op tot iets meer dan 1. Door te delen door die som tellen de drie kansen op tot 1.
    """
    ruw = 1 / df[schema.ODDS]
    return ruw["B365H"] / ruw.sum(axis=1)


def evalueer(pijplijn: Pipeline, train: pd.DataFrame, deel: pd.DataFrame, cfg: dict) -> dict:
    """Log loss van de drie modellen op één deel (validatie of test). Lager is beter.

    Alleen wedstrijden mét odds, zodat de drie modellen op exact dezelfde wedstrijden scoren.
    """
    deel = deel.dropna(subset=schema.ODDS)
    y = deel[cfg["doel"]]
    baseline1 = np.full(len(deel), train[cfg["doel"]].mean())
    return {
        "baseline 1 (altijd thuiswinst-% uit train)": log_loss(y, baseline1, labels=[0, 1]),
        "baseline 2 (Bet365)": log_loss(y, kans_bet365(deel), labels=[0, 1]),
        "logistische regressie": log_loss(
            y, pijplijn.predict_proba(deel[alle_features(cfg)])[:, 1], labels=[0, 1]
        ),
    }


def train(df: pd.DataFrame, cfg: dict) -> tuple[Pipeline, dict, dict]:
    """Train op de trainingsseizoenen, geef het model en de scores op validatie en test terug."""
    deel_train, validatie, test = splits(df, cfg)
    pijplijn = bouw_pijplijn(cfg).fit(deel_train[alle_features(cfg)], deel_train[cfg["doel"]])
    return (
        pijplijn,
        evalueer(pijplijn, deel_train, validatie, cfg),
        evalueer(pijplijn, deel_train, test, cfg),
    )


def main() -> None:
    cfg = laad_config()
    df = laad_en_maak_features(cfg)
    model, scores_validatie, scores_test = train(df, cfg)

    print(f"{'log loss (lager = beter)':45} {'validatie':>10} {'test':>10}")
    for naam in scores_validatie:
        print(f"{naam:45} {scores_validatie[naam]:10.3f} {scores_test[naam]:10.3f}")

    # We kiezen op validatie. De test kijken we alleen aan het einde, één keer.
    loss = scores_validatie["logistische regressie"]
    maximum = cfg["model"]["maximum_log_loss"]
    if loss > maximum:
        raise SystemExit(
            f"Log loss {loss:.3f} ligt boven het maximum {maximum}. Niet weggeschreven."
        )

    cfg["model"]["pad"].parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, cfg["model"]["pad"])
    print(f"Model weggeschreven naar {cfg['model']['pad']}")


if __name__ == "__main__":
    main()
