"""Test een model zonder te weten wat het exact moet voorspellen (les 5, slide 62-63).

`assert resultaat == verwacht` werkt hier niet: er is geen verwachte waarde. Wat je wél kan
vastleggen:

1. vorm en type van de output
2. invarianten: dingen die altijd waar moeten zijn
3. de pijplijn: ruwe invoer erin, voorspelling eruit, zonder te breken
4. persistentie: opgeslagen en teruggeladen geeft hetzelfde
5. reproduceerbaarheid: zelfde data, zelfde model
6. een ondergrens: slechter dan dit mag niet naar productie
"""

import joblib
import numpy as np

from src.predict import laad_model, voorspel
from src.train import train


def test_vorm_een_kans_per_wedstrijd(model, df, cfg):
    kansen = voorspel(model, df.head(50), cfg)
    assert kansen.shape == (50,)
    assert kansen.dtype == np.float64


def test_invariant_kansen_liggen_tussen_0_en_1(model, df, cfg):
    kansen = voorspel(model, df, cfg)
    assert np.all((kansen >= 0) & (kansen <= 1))


def test_geen_lege_voorspellingen(model, df, cfg):
    # Ook voor de eerste wedstrijd van een gepromoveerde ploeg, waar de vorm leeg is.
    assert not np.isnan(voorspel(model, df, cfg)).any()


def test_pijplijn_neemt_ruwe_rijen(model, df, cfg):
    # Een rij met lege vorm en een kolom die het model niet gebruikt. Het invullen moet
    # in de pijplijn zitten, niet in een notebookcel ervoor.
    rij = df[df["verschil_vorm_punten"].isna()].head(1).copy()
    rij["iets_onverwachts"] = "negeer mij"
    assert voorspel(model, rij, cfg).shape == (1,)


def test_persistentie(model, df, cfg, tmp_path):
    # Arrange: het model wegschrijven
    pad = tmp_path / "model.joblib"
    joblib.dump(model, pad)

    # Act: terugladen en voorspellen
    herladen = laad_model(pad)
    na = voorspel(herladen, df, cfg)

    # Assert: dezelfde voorspelling als ervoor
    np.testing.assert_allclose(na, voorspel(model, df, cfg))


def test_zelfde_data_zelfde_model(df, cfg):
    eerste, _, _ = train(df, cfg)
    tweede, _, _ = train(df, cfg)
    np.testing.assert_allclose(voorspel(eerste, df, cfg), voorspel(tweede, df, cfg))


def test_beter_dan_baseline_1(model_en_scores):
    # Wie altijd hetzelfde percentage zegt, verslaan we. Anders hebben de features geen nut.
    _, validatie, _ = model_en_scores
    assert (
        validatie["logistische regressie"] < validatie["baseline 1 (altijd thuiswinst-% uit train)"]
    )


def test_onder_het_maximum(model_en_scores, cfg):
    # Geen exacte waarde, een grens. Een test die faalt omdat de log loss van 0,632 naar
    # 0,633 gaat, gooit iedereen binnen de week weg.
    _, validatie, _ = model_en_scores
    assert validatie["logistische regressie"] <= cfg["model"]["maximum_log_loss"]
