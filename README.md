# 🚲 Bysykler Bergen – Prediksjon av ledige sykler

Prosjekt i INF161 ved Universitetet i Bergen (H25). Systemet predikerer antall ledige bysykler per stasjon én time frem i tid, basert på historiske stasjonsdata, turdata og værdata.

## Innhold

- [Oversikt](#oversikt)
- [Datakilder](#datakilder)
- [Features](#features)
- [Modeller](#modeller)
- [Installasjon](#installasjon)
- [Bruk](#bruk)
- [Prosjektstruktur](#prosjektstruktur)

## Oversikt

Bergen bysykkel har stasjoner spredt rundt i sentrum. For å kunne planlegge bedre – både for brukere og operatører – er det nyttig å vite hvor mange sykler som vil være tilgjengelige i nær fremtid. Denne løsningen bruker maskinlæring til å predikere antall ledige sykler **én time frem i tid** for ni utvalgte stasjoner.

**Stasjoner som dekkes:**
Møllendalsplass, Torgallmenningen, Grieghallen, Høyteknologisenteret, Studentboligene, Akvariet, Damsgårdsveien 71, Dreggsallmenningen Sør og Florida Bybanestopp.

## Datakilder

Rådataene ligger i `raw_data/` og består av tre CSV-filer:

| Fil | Beskrivelse |
|-----|-------------|
| `stations.csv` | Stasjonsdata med antall ledige sykler og plasser over tid |
| `trips.csv` | Individuelle turer med start-/sluttstasjon og tidspunkt |
| `weather.csv` | Værdata med temperatur, nedbør og vindstyrke |

> **Merk:** `trips.csv` (~155 MB) spores via [Git LFS](https://git-lfs.github.com). Sørg for at Git LFS er installert (`git lfs install`) før du kloner repoet, slik at filen lastes ned automatisk.

## Features

Modellen bruker følgende features for prediksjon:

- **Stasjonsdata:** `free_bikes`, `free_spots`, `fb_prev_h` (forrige times sykler)
- **Turdata:** `trips_in`, `trips_out` (antall turer inn/ut per time)
- **Værdata:** `temperature`, `precipitation`, `wind_speed`
- **Sykliske tidsfeatures:** sin/cos-transformasjoner av time, ukedag, måned og dag
- **Stasjonskoding:** One-hot encoding av stasjonsnavn

## Modeller

Følgende modeller trenes og sammenlignes automatisk i `train.py`:

- **Baseline:** DummyRegressor (gjennomsnitt og median)
- **ML-modeller:** Linear Regression, Lasso, Ridge, Random Forest, Gradient Boosting, KNN, SVR

Den beste modellen velges basert på RMSE på valideringssettet og lagres til `trained_model.pkl`. Dataen splittes 70/15/15 i trenings-, validerings- og testsett basert på kronologisk rekkefølge.

## Installasjon

**Krav:** Python 3.8+ og [Git LFS](https://git-lfs.github.com) for nedlasting av store datafiler.

```bash
git lfs install
git clone https://github.com/bragenybakk/INF161-prosjekt_BysyklerPrediksjon.git
cd INF161-prosjekt_BysyklerPrediksjon/prosjekt
pip install numpy pandas scikit-learn
```

## Bruk

### 1. Tren modellen

```bash
cd prosjekt
python train.py
```

Dette leser rådata, lager features, trener alle modeller, velger den beste og lagrer den til `trained_model.pkl`. Det genereres også en `model_ready.csv` med klargjort data.

### 2. Kjør prediksjon

```bash
python predict.py
```

Scriptet finner automatisk siste tilgjengelige tidsstempel i dataene, bygger features for neste hele klokketime, og skriver ut en tabell med nåværende og predikert antall sykler per stasjon. Manglende data håndteres med Last Observation Carried Forward (LOCF).

**Eksempel på output:**

```
Siste tidsstempel i data: 2024-10-15 14:32:00+02:00
Neste hele klokketime:    2024-10-15 15:00:00+02:00
Predikerer for:           2024-10-15 16:00:00+02:00

-------------------------------------------------------------
Stasjon                  Nåværende sykler Predikerte sykler
-------------------------------------------------------------
Møllendalsplass                         5                 7
Torgallmenningen                       12                10
...
-------------------------------------------------------------
```

## Prosjektstruktur

```
prosjekt/
├── raw_data/
│   ├── stations.csv
│   ├── trips.csv          # Spores via Git LFS (~155 MB)
│   └── weather.csv
├── train.py               # Trening og modellvalg
├── predict.py             # Prediksjon med trent modell
├── trained_model.pkl      # Lagret beste modell (genereres av train.py)
├── model_ready.csv        # Klargjort datasett (genereres av train.py)
└── hvordan_bruke.txt      # Bruksanvisning
```
