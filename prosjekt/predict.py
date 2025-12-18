import pickle
import numpy as np
import pandas as pd


# målstasjoner
stasjoner = [
    "Møllendalsplass",
    "Torgallmenningen",
    "Grieghallen",
    "Høyteknologisenteret",
    "Studentboligene",
    "Akvariet",
    "Damsgårdsveien 71",
    "Dreggsallmenningen Sør",
    "Florida Bybanestopp",
]

# featurekolonner
X_cols = [
    "free_bikes",
    "free_spots",
    "trips_in",
    "trips_out",
    "temperature",
    "precipitation",
    "wind_speed",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "month_sin",
    "month_cos",
    "day_sin",
    "day_cos",
    "fb_prev_h",
]

def read_raw():
    stations = pd.read_csv("raw_data/stations.csv")
    trips = pd.read_csv("raw_data/trips.csv")
    weather = pd.read_csv("raw_data/weather.csv")

    stations["timestamp"] = pd.to_datetime(stations["timestamp"], utc=True, errors="coerce")
    trips["started_at"] = pd.to_datetime(trips["started_at"], utc=True, errors="coerce")
    trips["ended_at"] = pd.to_datetime(trips["ended_at"], utc=True, errors="coerce")
    weather["timestamp"] = pd.to_datetime(weather["timestamp"], utc=True, errors="coerce")
    return stations, trips, weather

def latest_raw_timestamp(stations):
    return stations["timestamp"].max()

def current_bike_counts(stations):
    df = stations[stations["station"].isin(stasjoner)].copy()
    # Siste registrerte tidspunkt pr. stasjon
    last_ts = df.groupby("station")["timestamp"].max().reset_index()
    out = {}
    for _, row in last_ts.iterrows():
        st = row["station"]
        ts = row["timestamp"]
        val = df[(df["station"] == st) & (df["timestamp"] == ts)]["free_bikes"].iloc[-1]
        out[st] = int(val)
    return out

def stations_hourly(stations):
    df = stations[stations["station"].isin(stasjoner)].copy()
    df = df.drop(columns=["longitude", "latitude", "skipped_updates"], errors="ignore")
    df = df.sort_values(["station", "timestamp"]).set_index("timestamp")
    cols = ["free_bikes", "free_spots"]
    res = (
        df.groupby("station")[cols]
        .resample("h").last()
        .ffill()
        .reset_index()
        .sort_values(["station", "timestamp"])
    )
    return res


def ensure_station_rows_at_hour(s_hourly, feature_hour):
    frames = [s_hourly]
    for st in s_hourly["station"].unique():
        sub = s_hourly[s_hourly["station"] == st]
        last_before = sub[sub["timestamp"] <= feature_hour]
        if last_before.empty:
            continue
        last_row = last_before.iloc[-1]
        if pd.Timestamp(last_row["timestamp"]) < feature_hour:
            frames.append(pd.DataFrame({
                "station": [st],
                "timestamp": [feature_hour],
                "free_bikes": [last_row["free_bikes"]],
                "free_spots": [last_row["free_spots"]],
            }))
    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["station", "timestamp"]).reset_index(drop=True)
    return out

def trips_hourly(trips):
    df = trips.copy()
    df["started_h"] = df["started_at"].dt.floor("h")
    df["ended_h"] = df["ended_at"].dt.floor("h")

    starts = (
        df[df["start_station_name"].isin(stasjoner)]
        .groupby(["start_station_name", "started_h"]).size()
        .rename("trips_out")
        .reset_index()
        .rename(columns={"start_station_name": "station", "started_h": "timestamp"})
    )
    ends = (
        df[df["end_station_name"].isin(stasjoner)]
        .groupby(["end_station_name", "ended_h"]).size()
        .rename("trips_in")
        .reset_index()
        .rename(columns={"end_station_name": "station", "ended_h": "timestamp"})
    )

    hourly = pd.merge(starts, ends, on=["station", "timestamp"], how="outer").fillna(0)
    return hourly.sort_values(["timestamp"])  # per time

def weather_hourly(weather):
    df = weather.set_index("timestamp")
    return df.resample("h").mean().ffill().reset_index()

def merge_all(s_hourly, t_hourly, w_hourly):
    merged = s_hourly.merge(t_hourly, on=["timestamp", "station"], how="left")
    merged = merged.merge(w_hourly, on="timestamp", how="left")
    merged[["trips_in", "trips_out"]] = merged[["trips_in", "trips_out"]].fillna(0)
    return merged


def ensure_weather_row_at_hour(w_hourly, feature_hour):
    if (w_hourly["timestamp"] == feature_hour).any():
        return w_hourly
    before = w_hourly[w_hourly["timestamp"] <= feature_hour]
    if before.empty:
        return w_hourly
    last = before.iloc[-1].copy()
    last["timestamp"] = feature_hour
    return pd.concat([w_hourly, last.to_frame().T], ignore_index=True).sort_values("timestamp").reset_index(drop=True)

def create_features(df):
    # Sykliske features
    ts = df["timestamp"]
    df = df.copy()
    df["hour_sin"] = np.sin(2 * np.pi * ts.dt.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * ts.dt.hour / 24)
    df["weekday_sin"] = np.sin(2 * np.pi * ts.dt.weekday / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * ts.dt.weekday / 7)
    df["month_sin"] = np.sin(2 * np.pi * ts.dt.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * ts.dt.month / 12)
    df["day_sin"] = np.sin(2 * np.pi * ts.dt.day / 31)
    df["day_cos"] = np.cos(2 * np.pi * ts.dt.day / 31)

    # Forrige time
    df["fb_prev_h"] = df.groupby("station")["free_bikes"].shift(1)
    return df

def expected_station_cols():
    # Forsøk å lese forventede stasjonskolonner fra model_ready.csv.
    # Faller tilbake til å generere fra stasjoner med drop_first=True.
    try:
        ref = pd.read_csv("model_ready.csv", nrows=1)
        cols = [c for c in ref.columns if c.startswith("station_")]
        if cols:
            return cols
    except Exception:
        pass

    tmp = pd.DataFrame({"station": stasjoner})
    dummies = pd.get_dummies(tmp, columns=["station"], drop_first=True, dtype=int)
    return [c for c in dummies.columns if c.startswith("station_")]

def build_X_for_hour(merged, feature_hour):
    df = create_features(merged)

    # Kun rader med historikk (fb_prev_h)
    df = df.dropna(subset=["fb_prev_h"])  # behold kun rader med historikk

    # Rader for angitt time
    rows = df[df["timestamp"] == feature_hour].copy()
    if rows.empty:
        raise ValueError(f"Fant ikke data for timen {feature_hour}")

    # For utskrift senere
    rows_for_output = rows[["station", "free_bikes", "timestamp"]].copy()

    # One-hot for stasjon, samme som i train.py (drop_first=True)
    rows = pd.get_dummies(rows, columns=["station"], drop_first=True, dtype=int)

    # Sørg for riktige stasjonskolonner og rekkefølge
    station_cols = expected_station_cols()
    for col in station_cols:
        if col not in rows.columns:
            rows[col] = 0

    # Sørg for alle feature-kolonner
    for col in X_cols:
        if col not in rows.columns:
            rows[col] = 0

    # Reindekser til nøyaktig featurerekkefølge
    feature_cols = X_cols + station_cols
    rows = rows.reindex(columns=feature_cols, fill_value=0)

    return rows, rows_for_output

def main():
    # 1) Les rådata og finn siste tidsstempel (for utskrift)
    try:
        stations, trips, weather = read_raw()
    except FileNotFoundError:
        print("Feil: Kunne ikke lese rådata fra 'raw_data/'.")
        return

    # Siste ts hentes KUN fra stations (enkelt og i tråd med ønsket)
    last_raw_ts = latest_raw_timestamp(stations)
    next_whole_hour = last_raw_ts.ceil("h")
    predict_for = next_whole_hour + pd.Timedelta(hours=1)

    # 2) Bygg datagrunnlag til siste hele time
    s_hourly = stations_hourly(stations)
    # Utvid til å inkludere next_whole_hour for alle stasjoner (LOCF frem i tid)
    s_hourly = ensure_station_rows_at_hour(s_hourly, next_whole_hour)
    t_hourly = trips_hourly(trips)
    w_hourly = weather_hourly(weather)
    # Sørg for værverdier på next_whole_hour (LOCF)
    w_hourly = ensure_weather_row_at_hour(w_hourly, next_whole_hour)
    merged = merge_all(s_hourly, t_hourly, w_hourly)

    # 3) Lag X for "neste hele klokketime" (predikerer én time etter denne)
    try:
        X_pred, rows_out = build_X_for_hour(merged, next_whole_hour)
    except Exception as e:
        print(f"Feil ved dataklargjøring: {e}")
        return

    # 4) Last inn lagret beste modell
    try:
        with open("trained_model.pkl", "rb") as f:
            model = pickle.load(f)
    except FileNotFoundError:
        print("Feil: Fant ikke 'trained_models.pkl'. Kjør først 'python train.py'.")
        return

    # 5) Gjør prediksjoner
    preds = model.predict(X_pred)

    # 6) Hent nåværende sykler (siste registrerte måling per stasjon)
    now_counts = current_bike_counts(stations)

    # 7) Utskrift (UTC -> Europe/Oslo)
    bergen_tz = "Europe/Oslo"
    print(f"Siste tidsstempel i data: {last_raw_ts.tz_convert(bergen_tz)}")
    print(f"Neste hele klokketime: {next_whole_hour.tz_convert(bergen_tz)}")
    print(f"Predikerer for tidsstempel: {predict_for.tz_convert(bergen_tz)}")
    print()

    # Formater tabell
    width1, width2, width3 = 25, 18, 18
    print("-" * (width1 + width2 + width3))
    print(f"{'Stasjon':<{width1}}{'Nåværende sykler':>{width2}}{'Predikerte sykler':>{width3}}")
    print("-" * (width1 + width2 + width3))

    for i, row in rows_out.reset_index(drop=True).iterrows():
        st = row["station"]
        current_bikes = int(now_counts.get(st, 0))
        pred_bikes = int(round(float(preds[i])))
        print(f"{st:<{width1}}{current_bikes:>{width2}}{pred_bikes:>{width3}}")

    print("-" * (width1 + width2 + width3))


if __name__ == "__main__":
    main()
