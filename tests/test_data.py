"""Test de data, niet alleen de code. Garbage in, garbage out (les 5, slide 64).

Elke test neemt de echte data, maakt er bewust iets aan stuk, en controleert dat het schema
het tegenhoudt. Een schema dat alles doorlaat, is geen schema.
"""

import pandera.errors
import pytest

from src import schema


def test_echte_data_is_geldig(ruw):
    schema.GESPEELD.validate(ruw)


def test_onbekende_uitslag_wordt_geweigerd(ruw):
    kapot = ruw.copy()
    kapot.loc[0, "FTR"] = "X"
    with pytest.raises(pandera.errors.SchemaError):
        schema.GESPEELD.validate(kapot)


def test_uitslag_die_niet_klopt_met_de_score_wordt_geweigerd(ruw):
    # Het stilste soort fout: H, D en A zijn allemaal geldig, maar niet bij deze score.
    kapot = ruw.copy()
    kapot.loc[0, ["FTHG", "FTAG", "FTR"]] = [3, 0, "A"]
    with pytest.raises(pandera.errors.SchemaError):
        schema.GESPEELD.validate(kapot)


def test_odds_onder_1_worden_geweigerd(ruw):
    # Odds van 0,5 betekent: je krijgt minder terug dan je inzet, zelfs als je wint.
    kapot = ruw.assign(B365H=0.5)
    with pytest.raises(pandera.errors.SchemaError):
        schema.GESPEELD.validate(kapot)


def test_hernoemde_kolom_wordt_geweigerd(ruw):
    kapot = ruw.rename(columns={"FTHG": "thuisgoals"})
    with pytest.raises(pandera.errors.SchemaError):
        schema.GESPEELD.validate(kapot)


def test_dubbele_wedstrijd_wordt_geweigerd(ruw):
    kapot = ruw.copy()
    kapot.loc[1, ["Date", "HomeTeam", "AwayTeam"]] = kapot.loc[0, ["Date", "HomeTeam", "AwayTeam"]]
    with pytest.raises(pandera.errors.SchemaError):
        schema.GESPEELD.validate(kapot)


def test_komende_wedstrijd_vraagt_geen_uitslag(ruw):
    # Voor een wedstrijd die nog moet komen, ken je de uitslag niet. KOMEND mag die niet eisen.
    komend = ruw.dropna(subset=schema.ODDS)[
        ["Div", "Date", "Time", "HomeTeam", "AwayTeam", *schema.ODDS]
    ]
    schema.KOMEND.validate(komend)
