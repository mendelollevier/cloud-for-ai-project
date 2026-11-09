"""De winstcurve: hoeveel winst je maakt bij elke drempel (les 3, slide 15-18).

python -m src.winst

We geven een tip als de verwachte winst per ingezette euro groter is dan de drempel:

    verwachte winst = kans_model x odds - 1

Drempel 0 is het theoretische breekpunt uit deel 1 (kans > 1/odds). Een hogere drempel geeft
minder tips, maar alleen die waar het model het sterkst afwijkt van Bet365.
We kiezen de drempel op validatie (24-25) en kijken daarna één keer naar test (25-26).
"""

import numpy as np
import pandas as pd

from .config import laad_config
from .predict import voorspel
from .train import laad_en_maak_features, splits, train


def verwachte_winst(kans: np.ndarray, odds: np.ndarray) -> np.ndarray:
    """Verwachte winst per ingezette euro. Groter dan 0 = op lange termijn winst."""
    return kans * odds - 1


def winst(deel: pd.DataFrame, kansen: np.ndarray, drempel: float, inzet: float) -> dict:
    """Wat er gebeurd zou zijn als we op elke tip boven de drempel `inzet` euro hadden gezet."""
    tip = verwachte_winst(kansen, deel["B365H"].to_numpy()) > drempel
    gewonnen = deel["thuiswinst"].to_numpy()[tip] == 1
    odds = deel["B365H"].to_numpy()[tip]
    resultaat = np.where(gewonnen, inzet * (odds - 1), -inzet).sum()
    return {
        "drempel": drempel,
        "tips": int(tip.sum()),
        "winst": round(float(resultaat), 2),
        "rendement": round(float(resultaat) / (inzet * tip.sum()), 3) if tip.any() else 0.0,
    }


def winstcurve(model, deel: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Winst bij elke drempel uit de config. Alleen wedstrijden met odds: zonder odds geen inzet."""
    deel = deel.dropna(subset=["B365H"])
    kansen = voorspel(model, deel, cfg)
    w = cfg["winst"]
    return pd.DataFrame([winst(deel, kansen, d, w["inzet"]) for d in w["drempels"]])


def kies_drempel(curve: pd.DataFrame, minimum_tips: int) -> float:
    """De drempel met de meeste winst, alleen bij genoeg tips (anders kies je op geluk)."""
    genoeg = curve[curve["tips"] >= minimum_tips]
    return float(genoeg.loc[genoeg["winst"].idxmax(), "drempel"])


def main() -> None:
    cfg = laad_config()
    df = laad_en_maak_features(cfg)
    model, _, _ = train(df, cfg)
    _, validatie, test = splits(df, cfg)

    curve_validatie = winstcurve(model, validatie, cfg)
    seizoen, inzet = cfg["split"]["validatie"], cfg["winst"]["inzet"]
    print(f"Winstcurve op validatie ({seizoen}), {inzet} euro per tip:")
    print(curve_validatie.to_string(index=False))

    drempel = kies_drempel(curve_validatie, cfg["winst"]["minimum_tips"])
    curve_test = winstcurve(model, test, cfg)
    op_test = curve_test[curve_test["drempel"] == drempel].iloc[0]
    print(f"\nGekozen drempel op validatie: {drempel}")
    print(
        f"Op test ({cfg['split']['test']}): {int(op_test['tips'])} tips, "
        f"winst {op_test['winst']} euro, rendement {op_test['rendement']:.1%}"
    )


if __name__ == "__main__":
    main()
