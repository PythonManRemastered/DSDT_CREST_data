# ============================================================
# FACTORY PROXIMITY & AQI ANALYSIS (ROBUST VERSION)
# ============================================================

import pandas as pd
import numpy as np
import math
import os
from scipy import stats
import statsmodels.api as sm


# ============================================================
# LOAD DATA
# ============================================================

station_day = pd.read_csv("data/station_day.csv")
station_meta = pd.read_csv("data/dataset_stationID.csv")
factory_data = pd.read_csv("data/factory_loc.csv")


# ============================================================
# CLEAN COLUMN NAMES (ROBUST AGAINST SPACES / CASE ISSUES)
# ============================================================

station_day.columns = station_day.columns.str.strip().str.lower()
station_meta.columns = station_meta.columns.str.strip().str.lower()
factory_data.columns = factory_data.columns.str.strip().str.lower()


# ============================================================
# CLEAN STATION AQI DATA (2018–2020 ONLY)
# ============================================================

station_day = station_day[["stationid", "date", "aqi"]]
station_day = station_day.dropna(subset=["aqi"])

station_day["aqi"] = pd.to_numeric(station_day["aqi"], errors="coerce")
station_day["date"] = pd.to_datetime(station_day["date"])

station_day["year"] = station_day["date"].dt.year
station_day["month"] = station_day["date"].dt.month

# Restrict to 2018–2020
station_day = station_day[
    (station_day["year"] >= 2018) &
    (station_day["year"] <= 2020)
]


# ============================================================
# ASSIGN SEASONS (1–4)
# ============================================================

station_day["season"] = ((station_day["month"] - 1) // 3) + 1


# ============================================================
# COMPUTE SEASONAL MEAN AQI
# ============================================================

seasonal_stats = (
    station_day
    .groupby(["stationid", "year", "season"])["aqi"]
    .mean()
    .reset_index()
)

seasonal_stats.rename(columns={"aqi": "aqi_mean"}, inplace=True)


# ============================================================
# MERGE STATION COORDINATES
# ============================================================

station_meta = station_meta[["stationid", "latitude", "longitude"]]

seasonal_stats = seasonal_stats.merge(
    station_meta,
    on="stationid",
    how="left"
)

seasonal_stats = seasonal_stats.dropna(subset=["latitude", "longitude"])


# ============================================================
# CLEAN FACTORY DATA
# ============================================================

factory_data = factory_data[["poi_id", "latitude", "longitude"]]
factory_data = factory_data.dropna()

factory_data["latitude"] = pd.to_numeric(factory_data["latitude"])
factory_data["longitude"] = pd.to_numeric(factory_data["longitude"])


# ============================================================
# HAVERSINE FUNCTION
# ============================================================

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km

    lat1, lon1, lat2, lon2 = map(
        math.radians,
        [lat1, lon1, lat2, lon2]
    )

    a = (math.sin((lat2 - lat1) / 2) ** 2 +
         math.cos(lat1) * math.cos(lat2) *
         math.sin((lon2 - lon1) / 2) ** 2)

    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ============================================================
# COMPUTE FACTORY METRICS (50 KM CUTOFF)
# ============================================================

def compute_factory_metrics(station_lat, station_lon):

    distances = []

    for _, factory in factory_data.iterrows():
        d = haversine_distance(
            station_lat,
            station_lon,
            factory["latitude"],
            factory["longitude"]
        )
        distances.append(d)

    distances = np.array(distances)

    # Restrict to factories within 50 km
    local_factories = distances[distances <= 50]

    if len(local_factories) == 0:
        return np.nan, 0, 0

    closest_distance = np.min(local_factories)
    density_50km = len(local_factories)

    # Inverse-distance weighting only for local factories
    inverse_score = np.sum(1 / (local_factories + 1))

    return closest_distance, density_50km, inverse_score


closest_list = []
density_list = []
inverse_list = []

for _, row in seasonal_stats.iterrows():
    cd, dens, inv = compute_factory_metrics(
        row["latitude"],
        row["longitude"]
    )
    closest_list.append(cd)
    density_list.append(dens)
    inverse_list.append(inv)

seasonal_stats["closest_distance"] = closest_list
seasonal_stats["factory_density_50km"] = density_list
seasonal_stats["inverse_distance_score_50km"] = inverse_list

# Remove stations with no nearby factories
seasonal_stats = seasonal_stats.dropna(subset=["closest_distance"])

# Log transform (distance decay modelling)
seasonal_stats["log_closest_distance"] = np.log(
    seasonal_stats["closest_distance"] + 1
)


# ============================================================
# CORRELATION ANALYSIS
# ============================================================

results_text = []
results_text.append("=== FACTORY PROXIMITY & AQI ANALYSIS (2018–2020) ===\n")

variables = [
    "log_closest_distance",
    "factory_density_50km",
    "inverse_distance_score_50km"
]

for var in variables:
    r, p = stats.pearsonr(
        seasonal_stats[var],
        seasonal_stats["aqi_mean"]
    )
    results_text.append(f"\nPearson correlation (AQI vs {var})")
    results_text.append(f"r = {round(r, 5)}")
    results_text.append(f"p-value = {p}")


# ============================================================
# HYPOTHESIS TEST (LOG DISTANCE)
# ============================================================

r = stats.pearsonr(
    seasonal_stats["log_closest_distance"],
    seasonal_stats["aqi_mean"]
)[0]

n = len(seasonal_stats)

t_stat = (r * np.sqrt(n - 2)) / np.sqrt(1 - r**2)
p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-2))

results_text.append("\n=== Hypothesis Test (Log Closest Distance) ===")
results_text.append(f"t-statistic = {t_stat}")
results_text.append(f"p-value = {p_value}")


# ============================================================
# MULTIPLE LINEAR REGRESSION
# ============================================================

X = seasonal_stats[
    ["log_closest_distance",
     "factory_density_50km",
     "inverse_distance_score_50km"]
]

X = sm.add_constant(X)
y = seasonal_stats["aqi_mean"]

model = sm.OLS(y, X).fit()

results_text.append("\n=== MULTIPLE LINEAR REGRESSION SUMMARY ===\n")
results_text.append(str(model.summary()))


# ============================================================
# SAVE RESULTS
# ============================================================

folder_name = "Final Results"
os.makedirs(folder_name, exist_ok=True)

file_path = os.path.join(folder_name, "Results_Factory.txt")

with open(file_path, "w") as f:
    for line in results_text:
        f.write(line + "\n")

print("Analysis complete.")
print("Results saved to:", file_path)