"""Hoe de data eruit hoort te zien, vastgelegd met Pandera (les 5, slide 64 en 73).

Twee schema's, want op het moment van voorspellen ken je de uitslag nog niet:

- KOMEND: een wedstrijd die nog gespeeld moet worden, met de odds van Bet365
- GESPEELD: dezelfde kolommen plus uitslag en statistieken, om op te trainen

In GESPEELD mogen odds, ruststand en statistieken leeg zijn: een paar wedstrijden missen
ze. Die tellen dan niet mee in de berekening waarvoor ze nodig zijn.
"""

import numpy as np
import pandera.pandas as pa

ODDS = ["B365H", "B365D", "B365A"]
UITSLAG = ["H", "D", "A"]


def odds(leeg_mag: bool) -> dict:
    return {k: pa.Column(float, pa.Check.gt(1), nullable=leeg_mag) for k in ODDS}


def telling(maximum: int | None = None) -> pa.Column:
    """Een aantal (schoten, kaarten...). Float omdat een lege waarde in pandas een float is."""
    check = pa.Check.ge(0) if maximum is None else pa.Check.in_range(0, maximum)
    return pa.Column(float, check, nullable=True)


def uitslag(thuis, uit):
    """Welke uitslag hoort bij deze doelpunten: H, D of A."""
    return np.where(thuis > uit, "H", np.where(thuis == uit, "D", "A"))


def niet_leeg_of(df, kolommen, voorwaarde):
    """Een regel die alleen geldt als alle kolommen ingevuld zijn."""
    return voorwaarde | df[kolommen].isna().any(axis=1)


# Regels die in beide schema's gelden.
BASIS_KOLOMMEN = {
    "Div": pa.Column(str, pa.Check.eq("B1")),
    "Date": pa.Column(pa.DateTime),
    "Time": pa.Column(str, pa.Check.str_matches(r"^\d{2}:\d{2}$")),
    "HomeTeam": pa.Column(str, pa.Check.str_length(min_value=1)),
    "AwayTeam": pa.Column(str, pa.Check.str_length(min_value=1)),
}
BASIS_REGELS = [
    pa.Check(lambda df: df["HomeTeam"] != df["AwayTeam"], error="ploeg speelt tegen zichzelf"),
    # 1/H + 1/D + 1/A ligt net boven 1: dat surplus is de marge van de bookmaker.
    pa.Check(
        lambda df: niet_leeg_of(
            df, ODDS, (1 / df["B365H"] + 1 / df["B365D"] + 1 / df["B365A"]).between(1, 1.25)
        ),
        error="odds geven een onmogelijke marge",
    ),
]

KOMEND = pa.DataFrameSchema(
    {**BASIS_KOLOMMEN, **odds(leeg_mag=False)},
    checks=BASIS_REGELS,
    unique=["Date", "HomeTeam", "AwayTeam"],
    strict=False,  # een extra kolom is geen probleem, een ontbrekende wel
    name="komende wedstrijden",
)

GESPEELD = pa.DataFrameSchema(
    {
        **BASIS_KOLOMMEN,
        **odds(leeg_mag=True),
        "FTHG": pa.Column(int, pa.Check.in_range(0, 15)),
        "FTAG": pa.Column(int, pa.Check.in_range(0, 15)),
        "FTR": pa.Column(str, pa.Check.isin(UITSLAG)),
        "HTHG": telling(15),
        "HTAG": telling(15),
        "HTR": pa.Column(str, pa.Check.isin(UITSLAG), nullable=True),
        **{k: telling() for k in ["HS", "AS", "HST", "AST", "HF", "AF", "HC", "AC"]},
        "HY": telling(15),
        "AY": telling(15),
        "HR": telling(5),
        "AR": telling(5),
        "thuiswinst": pa.Column(int, pa.Check.isin([0, 1])),
    },
    checks=[
        *BASIS_REGELS,
        pa.Check(lambda df: df["FTR"] == uitslag(df["FTHG"], df["FTAG"]), error="FTR klopt niet"),
        pa.Check(
            lambda df: niet_leeg_of(
                df, ["HTHG", "HTAG"], (df["HTHG"] <= df["FTHG"]) & (df["HTAG"] <= df["FTAG"])
            ),
            error="ruststand hoger dan eindstand",
        ),
        pa.Check(
            lambda df: niet_leeg_of(
                df, ["HS", "HST", "AS", "AST"], (df["HST"] <= df["HS"]) & (df["AST"] <= df["AS"])
            ),
            error="meer schoten op doel dan schoten",
        ),
    ],
    unique=["Date", "HomeTeam", "AwayTeam"],
    strict=False,
    name="gespeelde wedstrijden",
)
