"""Test de winstberekening met een voorbeeld dat je met de hand kan narekenen."""

import numpy as np
import pandas as pd

from src.winst import winst


def test_winst_met_de_hand_nagerekend():
    # Drie wedstrijden, odds 2, 3 en 4. Het model geeft telkens 50% kans.
    # Verwachte winst: 0,5x2-1 = 0, 0,5x3-1 = 0,5, 0,5x4-1 = 1.
    # Bij drempel 0,2 tippen we dus alleen wedstrijd 2 en 3.
    # Wedstrijd 2 gewonnen: +10 x (3-1) = +20. Wedstrijd 3 verloren: -10. Samen +10.
    deel = pd.DataFrame({"B365H": [2.0, 3.0, 4.0], "thuiswinst": [1, 1, 0]})
    uitkomst = winst(deel, np.array([0.5, 0.5, 0.5]), drempel=0.2, inzet=10)
    assert uitkomst["tips"] == 2
    assert uitkomst["winst"] == 10


def test_hoge_drempel_geeft_geen_tips():
    deel = pd.DataFrame({"B365H": [2.0, 3.0], "thuiswinst": [1, 0]})
    uitkomst = winst(deel, np.array([0.5, 0.5]), drempel=10, inzet=10)
    assert uitkomst["tips"] == 0
    assert uitkomst["winst"] == 0
