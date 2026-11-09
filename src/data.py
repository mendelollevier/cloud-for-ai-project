"""Data inlezen, en ze meteen valideren. Wat niet klopt, komt hier niet voorbij."""

from pathlib import Path

import pandas as pd
import pandera.pandas as pa

from . import schema

# Enkel wat we nodig hebben. Alle odds-kolommen behalve Bet365 pre-closing laten we weg:
# de closing odds (met een C) zijn pas bij de aftrap bekend, dat zou een lek zijn.
KOLOMMEN = [
    "Div", "Date", "Time", "HomeTeam", "AwayTeam",
    "FTHG", "FTAG", "FTR", "HTHG", "HTAG", "HTR",
    "HS", "AS", "HST", "AST", "HF", "AF", "HC", "AC", "HY", "AY", "HR", "AR",
    "B365H", "B365D", "B365A",
]  # fmt: skip
# Football-data schrijft een ploeg soms anders.
# Eén naam per ploeg, anders lijkt het een nieuwe ploeg.
ZELFDE_PLOEG = {"KV Mechelen": "Mechelen"}


def laad_seizoenen(map_: str | Path) -> pd.DataFrame:
    """Lees alle seizoensbestanden uit een map in en zet ze onder elkaar, oudste eerst.

    Elk bestand heet naar zijn seizoen, bv. `21-22.csv`. Die naam wordt de kolom `seizoen`.
    """
    stukken = []
    for bestand in sorted(Path(map_).glob("*.csv")):
        # utf-8-sig: de bestanden beginnen met een onzichtbaar teken (BOM) voor "Div".
        df = pd.read_csv(bestand, encoding="utf-8-sig")
        df = df[[k for k in KOLOMMEN if k in df.columns]]
        df["seizoen"] = bestand.stem
        stukken.append(df)
    if not stukken:
        raise FileNotFoundError(f"Geen csv-bestanden in {map_}")
    alles = pd.concat(stukken, ignore_index=True)
    # dayfirst: 07/08/2026 is 7 augustus, niet 8 juli. Fout gelezen breekt de tijdsplit.
    alles["Date"] = pd.to_datetime(alles["Date"], format="%d/%m/%Y")
    # Doel: 1 = thuiswinst, 0 = gelijk of uitwinst. Nog niet gespeeld = leeg (NaN).
    alles["thuiswinst"] = alles["FTR"].map({"H": 1, "D": 0, "A": 0})
    alles[["HomeTeam", "AwayTeam"]] = alles[["HomeTeam", "AwayTeam"]].replace(ZELFDE_PLOEG)
    return alles.sort_values(["Date", "Time"], kind="stable").reset_index(drop=True)


def laad_data(
    pad: str | Path, valideer_met: pa.DataFrameSchema | None = schema.GESPEELD
) -> pd.DataFrame:
    """Lees een csv in en valideer hem tegen een schema.

    `lazy=True` verzamelt alle fouten in één keer, in plaats van te stoppen bij de eerste.
    Geef `valideer_met=None` om zonder validatie in te lezen. Dat doe je alleen om te zien
    wat er misloopt zonder.
    """
    df = pd.read_csv(pad)
    if valideer_met is None:
        return df
    try:
        return valideer_met.validate(df, lazy=True)
    except pa.errors.SchemaErrors as fouten:
        raise SystemExit(samenvatting(fouten, pad)) from None


def samenvatting(fouten: pa.errors.SchemaErrors, pad) -> str:
    """Eén regel per kapotte controle, met drie voorbeelden, in plaats van een muur tekst."""
    regels = [f"{pad} voldoet niet aan het schema:"]
    for (kolom, controle), groep in fouten.failure_cases.groupby(["column", "check"], dropna=False):
        voorbeelden = ", ".join(str(v) for v in groep["failure_case"].unique()[:3])
        regels.append(f"  {kolom}: {controle}, {len(groep)} rijen fout, bv. {voorbeelden}")
    return "\n".join(regels)
