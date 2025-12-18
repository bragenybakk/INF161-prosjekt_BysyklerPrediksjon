# Importerer nødvendige biblioteker

import numpy as np
import pandas as pd 
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import pickle

def clean_and_split_stations(s_df, stasjoner, train_end, val_end, overlap_start, overlap_end):
    # Filtrer stasjoner og dropp kolonner
    s_df_clean = s_df[s_df["station"].isin(stasjoner)]
    s_df_clean = s_df_clean.drop(columns=["longitude", "latitude", "skipped_updates"], errors="ignore")
    
    # Konverter til datetime
    s_df_clean["timestamp"] = pd.to_datetime(s_df_clean["timestamp"], utc=True, errors="coerce")
    
    # Filtrer til felles tidsområde FØRST
    s_df_clean = s_df_clean[
        (s_df_clean["timestamp"] >= overlap_start) & 
        (s_df_clean["timestamp"] <= overlap_end)
    ]
    
    # Split i train/val/test
    s_train = s_df_clean[s_df_clean['timestamp'] < train_end].copy()
    s_val = s_df_clean[(s_df_clean['timestamp'] >= train_end) & (s_df_clean['timestamp'] < val_end)].copy()
    s_test = s_df_clean[s_df_clean['timestamp'] >= val_end].copy()
    
    # Resample funksjon
    def resample_stations(df):
        df_sorted = df.sort_values(["station", "timestamp"])
        df_indexed = df_sorted.set_index("timestamp")
        cols = ["free_bikes", "free_spots"]
        df_resampled = (df_indexed.groupby("station")[cols]
                        .resample("h").last()
                        .ffill()
                        .reset_index()
                        .sort_values(["station", "timestamp"]))
        return df_resampled
    
    # Resample hver split
    s_hourly_train = resample_stations(s_train)
    s_hourly_val = resample_stations(s_val)
    s_hourly_test = resample_stations(s_test)
    
    return s_hourly_train, s_hourly_val, s_hourly_test


def clean_and_split_trips(t_df, stasjoner, train_end, val_end, overlap_start, overlap_end):
    # Dropp kolonner og konverter til datetime
    t_df_clean = t_df.drop(columns=["start_station_latitude", "start_station_longitude", 
                                     "end_station_latitude", "end_station_longitude"])
    t_df_clean["started_at"] = pd.to_datetime(t_df_clean["started_at"], utc=True, errors="coerce")
    t_df_clean["ended_at"] = pd.to_datetime(t_df_clean["ended_at"], utc=True, errors="coerce")
    
    # Filtrer til felles tidsområde FØRST
    t_df_clean = t_df_clean[
        (t_df_clean["started_at"] >= overlap_start) & 
        (t_df_clean["started_at"] <= overlap_end)
    ]
    
    # VIKTIG: Filtrer trips til kun de samme stasjonene som i stations-datasettet
    t_df_clean = t_df_clean[
        (t_df_clean["start_station_name"].isin(stasjoner)) | 
        (t_df_clean["end_station_name"].isin(stasjoner))
    ]
    
    # Split basert på started_at
    t_train = t_df_clean[t_df_clean['started_at'] < train_end].copy()
    t_val = t_df_clean[(t_df_clean['started_at'] >= train_end) & (t_df_clean['started_at'] < val_end)].copy()
    t_test = t_df_clean[t_df_clean['started_at'] >= val_end].copy()
    
    # Prosesser trips funksjon
    def process_trips(df):
        df["started_h"] = df["started_at"].dt.floor("h")
        df["ended_h"] = df["ended_at"].dt.floor("h")
        
        # Filtrer kun trips som starter eller slutter på våre stasjoner
        starts = (df[df["start_station_name"].isin(stasjoner)]
                    .groupby(["start_station_name","started_h"])
                    .size()
                    .rename("trips_out")
                    .reset_index()
                    .rename(columns={"start_station_name":"station","started_h":"timestamp"}))
        
        ends = (df[df["end_station_name"].isin(stasjoner)]
                  .groupby(["end_station_name", "ended_h"])
                  .size()
                  .rename("trips_in")
                  .reset_index()
                  .rename(columns={"end_station_name":"station","ended_h":"timestamp"}))
        
        hourly = pd.merge(starts, ends, on=["station", "timestamp"], how="outer").fillna(0)
        return hourly.sort_values(["timestamp"])
    
    # Prosesser hver split
    hourly_trips_train = process_trips(t_train)
    hourly_trips_val = process_trips(t_val)
    hourly_trips_test = process_trips(t_test)
    
    return hourly_trips_train, hourly_trips_val, hourly_trips_test


def clean_and_split_weather(w_df, train_end, val_end, overlap_start, overlap_end):
    # Konverter til datetime
    w_df["timestamp"] = pd.to_datetime(w_df["timestamp"], utc=True, errors="coerce")
    
    # Filtrer til felles tidsområde FØRST
    w_df = w_df[
        (w_df["timestamp"] >= overlap_start) & 
        (w_df["timestamp"] <= overlap_end)
    ]
    
    # Split weather
    w_train = w_df[w_df['timestamp'] < train_end].copy()
    w_val = w_df[(w_df['timestamp'] >= train_end) & (w_df['timestamp'] < val_end)].copy()
    w_test = w_df[w_df['timestamp'] >= val_end].copy()
    
    # Prosesser weather funksjon
    def process_weather(df):
        df_indexed = df.set_index("timestamp")
        w_resampled = df_indexed.resample("h").mean().ffill().reset_index()
        return w_resampled
    
    # Prosesser hver split
    w_hourly_train = process_weather(w_train)
    w_hourly_val = process_weather(w_val)
    w_hourly_test = process_weather(w_test)
    
    return w_hourly_train, w_hourly_val, w_hourly_test


def filter_cutoff_date(s_train, s_val, s_test, t_train, t_val, t_test, 
                       w_train, w_val, w_test, cutoff_date):
    s_train = s_train[s_train['timestamp'] >= cutoff_date]
    s_val = s_val[s_val['timestamp'] >= cutoff_date]
    s_test = s_test[s_test['timestamp'] >= cutoff_date]
    
    t_train = t_train[t_train['timestamp'] >= cutoff_date]
    t_val = t_val[t_val['timestamp'] >= cutoff_date]
    t_test = t_test[t_test['timestamp'] >= cutoff_date]
    
    w_train = w_train[w_train['timestamp'] >= cutoff_date]
    w_val = w_val[w_val['timestamp'] >= cutoff_date]
    w_test = w_test[w_test['timestamp'] >= cutoff_date]
    
    return s_train, s_val, s_test, t_train, t_val, t_test, w_train, w_val, w_test


def merge_datasets(s_hourly, hourly_trips, w_hourly):
    merged = s_hourly.merge(hourly_trips, on=["timestamp", "station"], how="left")
    merged = merged.merge(w_hourly, on="timestamp", how="left")
    merged[["trips_in","trips_out"]] = merged[["trips_in","trips_out"]].fillna(0)
    return merged


def create_features(df):
    df = df.sort_values(["station", "timestamp"])
    df["fb_next_h"] = df.groupby("station")["free_bikes"].shift(-1)
    df = df.dropna(subset=["fb_next_h"])

    # Sykliske features
    df['hour_sin'] = np.sin(2 * np.pi * df["timestamp"].dt.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df["timestamp"].dt.hour / 24)
    df['weekday_sin'] = np.sin(2 * np.pi * df["timestamp"].dt.weekday / 7)
    df['weekday_cos'] = np.cos(2 * np.pi * df["timestamp"].dt.weekday / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['timestamp'].dt.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['timestamp'].dt.month / 12)
    df['day_sin'] = np.sin(2 * np.pi * df['timestamp'].dt.day / 31)
    df['day_cos'] = np.cos(2 * np.pi * df['timestamp'].dt.day / 31)

    df["fb_prev_h"] = df.groupby("station")["free_bikes"].shift(1)
    df = df.dropna(subset=["fb_prev_h"])
    df = df.reset_index(drop=True)
    df = pd.get_dummies(df, columns=["station"], drop_first=True, dtype=int)
    return df

# Finn felles overlappende tidsperiode for alle datasett
def find_common_timerange(s_df, t_df, w_df):
    # Konverter til datetime
    s_temp = s_df.copy()
    t_temp = t_df.copy()
    w_temp = w_df.copy()
    
    s_temp["timestamp"] = pd.to_datetime(s_temp["timestamp"], utc=True, errors="coerce")
    t_temp["started_at"] = pd.to_datetime(t_temp["started_at"], utc=True, errors="coerce")
    w_temp["timestamp"] = pd.to_datetime(w_temp["timestamp"], utc=True, errors="coerce")
    
    # Finn min/max for hvert datasett
    stations_min, stations_max = s_temp["timestamp"].min(), s_temp["timestamp"].max()
    trips_min, trips_max = t_temp["started_at"].min(), t_temp["started_at"].max()
    weather_min, weather_max = w_temp["timestamp"].min(), w_temp["timestamp"].max()
    
    # Felles overlappende periode
    overlap_start = max(stations_min, trips_min, weather_min)
    overlap_end = min(stations_max, trips_max, weather_max)
    
    return overlap_start, overlap_end

# Les inn data
s_df = pd.read_csv("raw_data/stations.csv")
t_df = pd.read_csv("raw_data/trips.csv")
w_df = pd.read_csv("raw_data/weather.csv")

# Definer stasjoner
stasjoner = ["Møllendalsplass","Torgallmenningen","Grieghallen","Høyteknologisenteret",
             "Studentboligene","Akvariet","Damsgårdsveien 71","Dreggsallmenningen Sør",
             "Florida Bybanestopp"]

# Finn felles tidsområde
overlap_start, overlap_end = find_common_timerange(s_df, t_df, w_df)

# Beregn split-punkter basert på felles tidsområde
total_duration = overlap_end - overlap_start
train_end = overlap_start + total_duration * 0.70
val_end = overlap_start + total_duration * 0.85

# Rydd og split med felles tidsområde
s_train, s_val, s_test = clean_and_split_stations(s_df, stasjoner, train_end, val_end, overlap_start, overlap_end)
t_train, t_val, t_test = clean_and_split_trips(t_df, stasjoner, train_end, val_end, overlap_start, overlap_end)
w_train, w_val, w_test = clean_and_split_weather(w_df, train_end, val_end, overlap_start, overlap_end)

cutoff_date = pd.Timestamp("2024-08-22", tz="UTC")

# Filtrer bort data før cutoff_date
s_train, s_val, s_test, t_train, t_val, t_test, w_train, w_val, w_test = filter_cutoff_date(
    s_train, s_val, s_test, t_train, t_val, t_test, w_train, w_val, w_test, cutoff_date
)

# Merge
merged_train = merge_datasets(s_train, t_train, w_train)
merged_val = merge_datasets(s_val, t_val, w_val)
merged_test = merge_datasets(s_test, t_test, w_test)

# Features
final_train = create_features(merged_train)
final_val = create_features(merged_val)
final_test = create_features(merged_test)

# Sort only by timestamp
final_train = final_train.sort_values("timestamp")
final_val = final_val.sort_values("timestamp")
final_test = final_test.sort_values("timestamp")

# Kombiner alle final datasett til én fil
combined_data = pd.concat([final_train, final_val, final_test], ignore_index=True)

# Sorter etter timestamp for konsistens
combined_data = combined_data.sort_values('timestamp').reset_index(drop=True)

# Lagre til CSV
combined_data.to_csv('model_ready.csv', index=False)

print(f"\nKombinert datasett lagret som 'model_ready.csv' \n")

# MODELLERING

target = "fb_next_h" # target verdien
X_cols = [
    'free_bikes',    
    'free_spots',    
    'trips_in',      
    'trips_out',
    'temperature',   
    'precipitation',
    'wind_speed',
    'hour_sin',      
    'hour_cos',
    'weekday_sin',
    'weekday_cos',
    'month_sin',
    'month_cos',
    'day_sin',
    'day_cos',
    'fb_prev_h'      
]
station_cols = [col for col in combined_data.columns if col.startswith('station_')]

X = combined_data[X_cols + station_cols]
y = combined_data[target]

# Split X og y tilbake til train/val/test basert på antall rader
train_size = len(final_train)
val_size = len(final_val)

X_train = X[:train_size]
X_val = X[train_size:train_size + val_size]
X_test = X[train_size + val_size:]

y_train = y[:train_size]
y_val = y[train_size:train_size + val_size]
y_test = y[train_size + val_size:]

# trener baseline modeller

baseline_models = {"Dummy_mean" : DummyRegressor(strategy="mean"),
                "Dummy_median" : DummyRegressor(strategy="median"),}

for name, model in baseline_models.items():
    model.fit(X_train, y_train)
    y_val_pred = model.predict(X_val)
    rmse_model = np.sqrt(mean_squared_error(y_val, y_val_pred))
    print(f"{name} : med en RMSE på {rmse_model:.3f}")

# trener andre modeller

ml_models = {
    "Linear": LinearRegression(),
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
    "GradientBoosting": GradientBoostingRegressor(random_state=42),
    "KNN": KNeighborsRegressor(n_neighbors=10),
    "SVR": SVR(kernel='rbf'),
    "Lasso": Lasso(alpha=0.1),
    "Ridge": Ridge(alpha=1.0)
}

results = {} # lager en dic for rmse resultatene til modellene

for name, model in ml_models.items():
    print(f"Trener {name}...")
    model.fit(X_train, y_train)
    y_val_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    results[name] = rmse
    print(f"{name}: Val RMSE = {rmse:.3f}")

# finner beste modellen
best_model = min(results, key=results.get)
best_rmse = results[best_model]
print(f"Beste modellen er {best_model} med en RMSE = {best_rmse:.3f}")

train_results = {} # dic for rmse trenings resultater

for name, model in ml_models.items():
    y_train_pred = model.predict(X_train)
    rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_results[name] = rmse

print()

# sammeligner de to settene av resultater

for name in results.keys():
    train_rmse = train_results[name]
    val_rmse = results[name]
    diff = val_rmse - train_rmse

    print(f"{name}:(Diff: {diff:.3f})")

print()

# Tester ulike n-verdier for KNN

n_values = [1, 3, 5, 7, 10, 15, 20, 25, 30]
knn_results = {}

for n in n_values:
    knn_model = KNeighborsRegressor(n_neighbors=n)
    knn_model.fit(X_train, y_train)
    
    # Evaluer på både train og validation
    y_train_pred = knn_model.predict(X_train)
    y_val_pred = knn_model.predict(X_val)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    
    knn_results[n] = {'train': train_rmse, 'val': val_rmse, 'diff': val_rmse - train_rmse}
    
    print(f"n={n}: Train={train_rmse:.3f}, Val={val_rmse:.3f}, Diff={val_rmse - train_rmse:.3f}")


# Bruk beste modell for final test
best_model_obj = ml_models[best_model]
y_final_pred = best_model_obj.predict(X_test)

# Lagre den beste modellen til fil
with open('trained_model.pkl', 'wb') as f:
    pickle.dump(best_model_obj, f)

print(f"\n Beste modell lagret til 'trained_model.pkl'")

test_rmse = np.sqrt(mean_squared_error(y_test, y_final_pred))
print(f"\n Final test RMSE med {best_model}: {test_rmse:.3f}")