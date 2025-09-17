# pipeline.py
# Input: stations.csv, trips.csv, weather.csv
# Output: model_ready.csv

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

np.random.seed(100)

# LESER DATAEN

s_df = pd.read_csv("raw_data/stations.csv")
t_df = pd.read_csv("raw_data/trips.csv")
w_df = pd.read_csv("raw_data/weather.csv")

# Videre prøver jeg å rydde datasettet "stations" slik at den dataen jeg ikke trenger fjernes, og sorterer opp i datasettet

stasjoner = [
    "Møllendalsplass","Torgallmenningen","Grieghallen","Høyteknologisenteret",
    "Studentboligene","Akvariet","Damsgårdsveien 71","Dreggsallmenningen Sør",
    "Florida Bybanestopp"
]

# Filtrerer først, så dropper jeg ubrukte kolonner
s_df_clean = s_df[s_df["station"].isin(stasjoner)]
s_df_clean = s_df_clean.drop(columns=["longitude", "latitude", "skipped_updates"])

# Sørger for datetime i UTC
s_df_clean["timestamp"] = pd.to_datetime(s_df_clean["timestamp"], utc=True, errors="coerce")

# Sorterer og endrer index
s_sorted  = s_df_clean.sort_values(["station", "timestamp"])
s_indexed = s_sorted.set_index("timestamp")

# Resample til hver time pr stasjon, og fyll framover (LOCF) for relevante kolonner
cols = ["free_bikes", "free_spots"]  # tidsseriene vi vil resample
s_resampled = s_indexed.groupby("station")[cols].resample("h").ffill()

# Tilbake til vanlig index
s_hourly = s_resampled.reset_index()

# Rydder trips

# dropper unødvendige kolonner, og setter tid til datetime utc
t_df_clean = t_df.drop(columns=["start_station_latitude", "start_station_longitude", "end_station_latitude", "end_station_longitude"])
t_df_clean["started_at"] = pd.to_datetime(t_df_clean["started_at"], utc=True, errors="coerce")
t_df_clean["ended_at"] = pd.to_datetime(t_df_clean["ended_at"], utc=True, errors="coerce")

# lager times-gulv
t_df_clean["started_h"] = t_df_clean["started_at"].dt.floor("h")
t_df_clean["ended_h"] = t_df_clean["ended_at"].dt.floor("h")

# Antall turer som starter per stasjon/time
starts = (t_df_clean.groupby(["start_station_name","started_h"])
            .size()
            .rename("trips_out")
            .reset_index()
            .rename(columns={"start_station_name":"station","started_h":"timestamp"}))

# turer som slutter per stasjon/time
ends = (t_df_clean.groupby(["end_station_name", "ended_h"]).size().rename("trips_in").reset_index().rename(columns={"end_station_name":"station","ended_h":"timestamp"}))

# slår sammen turer ut og inn
hourly_trips = pd.merge(starts, ends, on=["station", "timestamp"], how="outer").fillna(0)

# Rydder weather

w_clean = w_df.copy() # lager kopi for å ikke tukle med originalen

w_clean["timestamp"] = pd.to_datetime(w_df["timestamp"], utc=True, errors="coerce")

w_clean = w_clean.set_index("timestamp")

# Times-oppløsning, gjennomsnitt per time
w_hourly = w_clean.resample("h").mean().reset_index()

# Merger sammen daten til et felles datasett

merged = s_hourly.merge(hourly_trips, on=["timestamp", "station"], how="left")
merged = merged.merge(w_hourly, on="timestamp", how="left")
merged[["trips_in","trips_out"]] = merged[["trips_in","trips_out"]].fillna(0)

# Ferdig data i nye csv

merged = merged.sort_values(["station","timestamp"])
merged["fb_next_h"] = merged.groupby("station")["free_bikes"].shift(-1)
merged = merged.dropna(subset=["fb_next_h"])

merged["hour"] = merged["timestamp"].dt.hour
merged["weekday"] = merged["timestamp"].dt.weekday
merged["fb_prev_h"] = merged.groupby("station")["free_bikes"].shift(1)

merged = merged.sort_values("timestamp").reset_index(drop=True) # Sorterer til slutt all dataen på tid, så den er klar

merged.to_csv("model_ready.csv", index=False)