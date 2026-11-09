"""Features maken: de vorm van beide ploegen, enkel uit wedstrijden van vóór de aftrap."""

import numpy as np
import pandas as pd

GEWICHTEN = np.array([1] * 20 + [2] * 5 + [3] * 4 + [5])
VORMKOLOMMEN = ["vorm_punten", "vorm_voor", "vorm_tegen", "vorm_schoten", "aantal_vorige"]


def gewogen(reeks):
    """Gewogen gemiddelde van de vorige wedstrijden. Lege waarden tellen niet mee."""
    w = GEWICHTEN[-len(reeks) :]
    ok = ~np.isnan(reeks)
    if not ok.any():
        return np.nan
    return np.average(reeks[ok], weights=w[ok])


def maak_features(df):
    """Voeg per wedstrijd het vormverschil toe: thuisploeg min uitploeg."""
    df = df.copy()

    # 1. Eén rij per ploeg per wedstrijd, thuis en uit samen.
    thuis = pd.DataFrame(
        {
            "wedstrijd": df.index,
            "Date": df["Date"],
            "ploeg": df["HomeTeam"],
            "voor": df["FTHG"],
            "tegen": df["FTAG"],
            "schoten": df["HS"],
            "punten": df["FTR"].map({"H": 3, "D": 1, "A": 0}),
        }
    )
    uit = pd.DataFrame(
        {
            "wedstrijd": df.index,
            "Date": df["Date"],
            "ploeg": df["AwayTeam"],
            "voor": df["FTAG"],
            "tegen": df["FTHG"],
            "schoten": df["AS"],
            "punten": df["FTR"].map({"H": 0, "D": 1, "A": 3}),
        }
    )
    per_ploeg = pd.concat([thuis, uit]).sort_values(["ploeg", "Date"])

    # 2. Gewogen vorm. shift(1): de wedstrijd zelf telt niet mee, anders is het een lek.
    for kolom in ["punten", "voor", "tegen", "schoten"]:
        per_ploeg[f"vorm_{kolom}"] = per_ploeg.groupby("ploeg")[kolom].transform(
            lambda s: s.shift(1).rolling(30, min_periods=1).apply(gewogen, raw=True)
        )

    # 3. Hoeveel vorige matchen de vorm draagt. Meer dan 30 zegt niets extra.
    per_ploeg["aantal_vorige"] = per_ploeg.groupby("ploeg").cumcount().clip(upper=30)

    # 4. Terug naar één rij per wedstrijd: thuisploeg min uitploeg.
    vorm = per_ploeg.set_index(["wedstrijd", "ploeg"])
    thuisvorm = vorm.loc[list(zip(df.index, df["HomeTeam"], strict=True)), VORMKOLOMMEN].to_numpy()
    uitvorm = vorm.loc[list(zip(df.index, df["AwayTeam"], strict=True)), VORMKOLOMMEN].to_numpy()
    verschil = pd.DataFrame(
        thuisvorm - uitvorm, index=df.index, columns=[f"verschil_{k}" for k in VORMKOLOMMEN]
    )
    return pd.concat([df, verschil], axis=1)
