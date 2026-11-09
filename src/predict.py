"""Een opgeslagen model laden en de kans op thuiswinst voorspellen.

python -m src.predict
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from .config import alle_features, laad_config
from .train import kans_bet365, laad_en_maak_features


def laad_model(pad: str | Path) -> Pipeline:
    return joblib.load(pad)


def voorspel(model: Pipeline, df: pd.DataFrame, cfg: dict) -> np.ndarray:
    """De kans op thuiswinst, één getal tussen 0 en 1 per wedstrijd."""
    return model.predict_proba(df[alle_features(cfg)])[:, 1]


def main() -> None:
    cfg = laad_config()
    df = laad_en_maak_features(cfg)
    live = df[df["seizoen"] == cfg["split"]["live"]].copy()
    live["kans_model"] = voorspel(laad_model(cfg["model"]["pad"]), live, cfg)
    live["kans_bet365"] = kans_bet365(live)

    print(f"{len(live)} wedstrijden in seizoen {cfg['split']['live']}.")
    kolommen = ["Date", "HomeTeam", "AwayTeam", "B365H", "kans_model", "kans_bet365"]
    print(live[kolommen].tail(10).round({"kans_model": 3, "kans_bet365": 3}).to_string(index=False))


if __name__ == "__main__":
    main()
