"""Test dat de features niet in de toekomst kijken (les 4, slide 28-36: data leakage)."""

import numpy as np
import pandas as pd

from src.features import maak_features


def test_geen_lek_uit_de_wedstrijd_zelf_of_later(ruw):
    # Verander de uitslag van één wedstrijd middenin. Als de features eerlijk zijn, mag dat
    # niets veranderen aan die wedstrijd zelf, en ook niet aan alle wedstrijden ervoor.
    # Alleen latere wedstrijden mogen die uitslag gebruiken.
    midden = ruw.index[len(ruw) // 2]
    aangepast = ruw.copy()
    aangepast.loc[midden, ["FTHG", "FTAG", "FTR"]] = [9, 0, "H"]

    voor = maak_features(ruw)
    na = maak_features(aangepast)

    kolommen = [k for k in voor.columns if k.startswith("verschil_")]
    pd.testing.assert_frame_equal(voor.loc[:midden, kolommen], na.loc[:midden, kolommen])


def test_eerste_wedstrijd_heeft_geen_vorm(ruw):
    # Op speeldag 1 heeft nog geen enkele ploeg een vorige wedstrijd: de vorm moet leeg zijn,
    # niet stiekem ingevuld met de uitslag van die wedstrijd zelf.
    df = maak_features(ruw)
    assert np.isnan(df["verschil_vorm_punten"].iloc[0])
