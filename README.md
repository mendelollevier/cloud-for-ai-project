# Cloud for AI-project: inzetten op thuiswinst?
Schoolproject voor het vak Cloud for AI.

Het model voorspelt voor elke wedstrijd in de Belgische Pro League hoe groot de kans is dat de thuisploeg wint, als een score tussen 0 en 1. Met die score en de odds van de bookmaker beslis je of inzetten de moeite loont.

## Data
Bron: football-data.co.uk, gratis CSV-bestanden per seizoen
Competitie: Belgische Pro League (B1.csv)
Uitleg van de kolommen: notes.txt

## Aanpak
ML-taak: binaire classificatie (thuiswinst ja/nee)
Beslissing: inzetten als de score hoger is dan 1/odds

## Metrieken:
ML: log loss
Software: % tips op tijd en up-to-date (ten laatste 24u voor de aftrap, met de uitslagen van de vorige speeldag)
Business: winst in € bij €10 per tip, vergeleken met niet inzetten

## Draaien

Alles vanuit de hoofdmap van de repo (niet vanuit `src/`).

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1      # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

pytest                             # tests
ruff check . ; ruff format --check .
python -m src.train                # traint en schrijft models/model.joblib
```

## Wat waar staat

```
configs/default.yaml   paden, features, seed. Geen code
src/                   de code: config, schema (Pandera), data, train, predict
tests/                 tests op de data en het model
data/                  de datasets
notebooks/             data-exploratie
models/                wordt aangemaakt door src.train, niet in git
```

## Planning
 Deel 1: projectvoorstel
 Deel 2: data-exploratie en twee modellen
 Deel 3: online zetten op een publieke URL

## Auteur
Mendel Ollevier
