# Cloud for AI-project: inzetten op thuiswinst?

Schoolproject voor het vak Cloud for AI.

Het model voorspelt voor elke wedstrijd in de Belgische Pro League hoe groot de kans is dat de
thuisploeg wint, als een getal tussen 0 en 1. Met die kans en de odds van Bet365 beslis je of
inzetten de moeite loont.

## Snel starten

Alles vanuit de hoofdmap van de repo (niet vanuit `src/`). Getest met Python 3.13 op Windows.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1      # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

pytest                             # 19 tests: data, features, model, winst
ruff check . ; ruff format --check .
python -m src.train                # traint, vergelijkt 3 modellen, schrijft models/model.joblib
python -m src.winst                # winstcurve: kiest de drempel op validatie, toont het resultaat op test
python -m src.predict              # kansen voor het lopende seizoen (eerst src.train draaien)
```

De data zit al in de repo (`data/seizoenen/`), je hoeft niets te downloaden.

## Data

- Bron: [football-data.co.uk](https://www.football-data.co.uk/belgiumm.php), gratis CSV per seizoen (B1.csv)
- Seizoenen 2021-22 tot en met 2025-26, plus het lopende seizoen 2026-27
- Eén bestand per seizoen in `data/seizoenen/`, genoemd naar het seizoen (bv. `24-25.csv`)
- Uitleg van de kolommen: `notes.txt` op football-data.co.uk

Wat er stuk was en wat ik eraan deed (zie `notebooks/exploratie.ipynb`):

- De bestanden beginnen met een onzichtbaar BOM-teken: inlezen met `utf-8-sig`
- Datums staan als dag/maand/jaar: expliciet `%d/%m/%Y`, anders breekt de tijdsplit
- Een paar wedstrijden zonder odds, ruststand of statistieken: die tellen niet mee waar ze nodig zijn
- "KV Mechelen" en "Mechelen" zijn dezelfde ploeg: samengevoegd in `src/data.py`
- Closing odds (kolommen met een C) zijn pas bij de aftrap bekend: weggelaten (lek)

## Aanpak

- ML-taak: binaire classificatie (thuiswinst ja/nee), doel `thuiswinst` uit `FTR`
- Features: gewogen vorm van beide ploegen over hun vorige 30 wedstrijden (punten, goals voor,
  goals tegen, schoten), thuisploeg min uitploeg. Alleen wedstrijden van vóór de aftrap (`shift(1)`)
- Split in de tijd: train 21-22 t/m 23-24, validatie 24-25, test 25-26
- Beslissing: tip als `kans × odds − 1` groter is dan een drempel, gekozen met de winstcurve

## Metrieken

- ML: log loss (lager = beter)
- Software: % tips op tijd en up-to-date (ten laatste 24u voor de aftrap, met de uitslagen van de vorige speeldag)
- Business: winst in € bij €10 per tip, vergeleken met niet inzetten

## Resultaten

| Model (log loss, lager = beter)            | Validatie 24-25 | Test 25-26 |
|--------------------------------------------|-----------------|------------|
| Baseline 1: altijd thuiswinst-% uit train  | 0,686           | 0,675      |
| Baseline 2: kans volgens Bet365            | 0,611           | 0,615      |
| Logistische regressie op vorm              | 0,632           | 0,637      |

Het model verslaat een vast percentage, maar niet Bet365. De winstcurve kiest op validatie
drempel 0,25 (27 tips, +31 euro), maar op test geeft die drempel 12 tips en −24 euro.
Er is dus geen betrouwbare winst tegen de bookmaker.

## Wat waar staat

```
configs/default.yaml   paden, features, split, grenzen. Geen code
src/data.py            seizoenen inlezen en opkuisen
src/schema.py          Pandera-schema's: GESPEELD (trainen) en KOMEND (voorspellen)
src/features.py        vorm-features, zonder in de toekomst te kijken
src/train.py           split, 2 baselines + logistische regressie, log loss
src/winst.py           winstcurve en drempel
src/predict.py         model laden en voorspellen
tests/                 tests op data, features, model en winst
data/seizoenen/        de datasets
notebooks/             data-exploratie
models/                wordt aangemaakt door src.train, niet in git
```

## Planning

- Deel 1: projectvoorstel
- Deel 2: data-exploratie en modellen vergelijken
- Deel 3: online zetten op een publieke URL

## Auteur

Mendel Ollevier
