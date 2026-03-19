# ============================================================
# FARM LOCATIONS VS AQI ANALYSIS
# AQI segmented by season, farms treated as static
# ============================================================

# ------------------------------------------------------------
# WHERE TO ADD PATHS
# ------------------------------------------------------------
# Replace these with full paths if needed.
# Example:
# station_day_path = "/Users/yourname/Desktop/DSDT_CREST/station_day.csv"
# ------------------------------------------------------------

station_day_path = "Data/station_day.csv"
station_meta_path = "Data/dataset_stationID.csv"
farm_data_path = "Data/farms_dataset.csv"

# ============================================================
# IMPORTS
# ============================================================

import os
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

# ============================================================
# PARAMETERS
# ============================================================

YEAR_START = 2018
YEAR_END = 2020

# Radius used for local farm density
DENSITY_RADIUS_KM = 100

# Number of nearest farms to use
K_NEAREST = 3

OUTPUT_FOLDER = "Final_Results"
OUTPUT_TEXT_FILE = "Results_Farm.txt"
OUTPUT_DATA_FILE = "Farm_Analysis_Dataset.csv"

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_columns(df):
    df.columns = [str(col).strip().lower() for col in df.columns]
    return df


def validate_required_columns(df, required_cols, df_name):
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise KeyError(
            f"{df_name} is missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )


def assign_season(month):
    # Season 1 = Jan-Mar, 2 = Apr-Jun, 3 = Jul-Sep, 4 = Oct-Dec
    return ((month - 1) // 3) + 1


def haversine_matrix(station_lats, station_lons, target_lats, target_lons):
    """
    Vectorized haversine distance matrix.
    Output shape: (number_of_stations, number_of_targets)
    Distances in km.
    """
    R = 6371.0

    station_lats_rad = np.radians(station_lats)[:, None]
    station_lons_rad = np.radians(station_lons)[:, None]
    target_lats_rad = np.radians(target_lats)[None, :]
    target_lons_rad = np.radians(target_lons)[None, :]

    dlat = target_lats_rad - station_lats_rad
    dlon = target_lons_rad - station_lons_rad

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(station_lats_rad) * np.cos(target_lats_rad) * np.sin(dlon / 2.0) ** 2
    )

    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return R * c


def safe_pearson(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan, np.nan

    return stats.pearsonr(x, y)


def safe_spearman(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan, np.nan

    return stats.spearmanr(x, y)


def pearson_t_test_from_r(r, n):
    if pd.isna(r) or n < 3 or abs(r) >= 1:
        return np.nan, np.nan

    t_stat = (r * np.sqrt(n - 2)) / np.sqrt(1 - r**2)
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n - 2))
    return t_stat, p_value


def add_header(results, title):
    results.append("\n" + "=" * 90)
    results.append(title)
    results.append("=" * 90)


def add_corr_results(results, df, x_col, y_col, label):
    x = df[x_col].to_numpy(dtype=float)
    y = df[y_col].to_numpy(dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    n = len(x)
    pearson_r, pearson_p = safe_pearson(x, y)
    spearman_rho, spearman_p = safe_spearman(x, y)
    t_stat, t_p = pearson_t_test_from_r(pearson_r, n)

    results.append(f"\n{label}")
    results.append(f"Observations: {n}")
    results.append(f"Pearson r: {pearson_r}")
    results.append(f"Pearson p-value: {pearson_p}")
    results.append(f"Pearson r^2: {pearson_r**2 if pd.notna(pearson_r) else np.nan}")
    results.append(f"Spearman rho: {spearman_rho}")
    results.append(f"Spearman p-value: {spearman_p}")
    results.append(f"Formal t-statistic: {t_stat}")
    results.append(f"Formal t-test p-value: {t_p}")


def run_hc3_regression(df, response_col, predictor_cols, include_season_dummies=False):
    try:
        work_df = df.copy()

        cols_needed = [response_col] + predictor_cols
        if include_season_dummies:
            cols_needed += ["season"]

        work_df = work_df[cols_needed].dropna().copy()

        if len(work_df) < 8:
            return "Not enough valid observations for regression."

        if include_season_dummies:
            season_dummies = pd.get_dummies(work_df["season"], prefix="season", drop_first=True)
            work_df = pd.concat([work_df.drop(columns=["season"]), season_dummies], axis=1)

        y = work_df[response_col].astype(float)
        X = work_df.drop(columns=[response_col]).astype(float)

        zero_var_cols = [col for col in X.columns if X[col].std() == 0]
        if zero_var_cols:
            X = X.drop(columns=zero_var_cols)

        if X.shape[1] == 0:
            return "Regression could not run because all predictors had zero variance."

        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit(cov_type="HC3")
        return str(model.summary())

    except Exception as e:
        return f"Regression failed: {e}"


def detect_farm_size_column(df):
    """
    Detect a plausible farm size column if one exists.
    """
    candidate_names = [
        "size",
        "farm_size",
        "farm size",
        "area",
        "farm_area",
        "farm area",
        "hectares",
        "hectare",
        "acreage",
        "acres",
        "acre",
        "area_ha",
        "area_hectare",
        "land_area",
        "land area"
    ]

    for col in df.columns:
        if col in candidate_names:
            return col

    for col in df.columns:
        lowered = str(col).lower()
        if any(token in lowered for token in ["size", "area", "hectare", "acre"]):
            return col

    return None


def infer_crop_count(crop_value):
    """
    Count listed crops in the crop column as a descriptive variable only.
    Not used as true size.
    """
    if pd.isna(crop_value):
        return np.nan
    parts = [p.strip() for p in str(crop_value).split(",") if p.strip()]
    return len(parts)

# ============================================================
# LOAD DATA
# ============================================================

station_day = pd.read_csv(station_day_path)
station_meta = pd.read_csv(station_meta_path)
farm_data = pd.read_csv(farm_data_path)

# ============================================================
# CLEAN COLUMN NAMES
# ============================================================

station_day = clean_columns(station_day)
station_meta = clean_columns(station_meta)
farm_data = clean_columns(farm_data)

# ============================================================
# VALIDATE REQUIRED COLUMNS
# ============================================================

validate_required_columns(
    station_day,
    ["stationid", "date", "aqi"],
    "station_day"
)

validate_required_columns(
    station_meta,
    ["stationid", "latitude", "longitude"],
    "dataset_stationID"
)

validate_required_columns(
    farm_data,
    ["latitude", "longitude", "country"],
    "farm dataset"
)

# ============================================================
# CLEAN AQI DATA
# ============================================================

station_day = station_day[["stationid", "date", "aqi"]].copy()
station_day["aqi"] = pd.to_numeric(station_day["aqi"], errors="coerce")
station_day["date"] = pd.to_datetime(station_day["date"], errors="coerce")
station_day = station_day.dropna(subset=["aqi", "date"]).copy()

station_day["year"] = station_day["date"].dt.year
station_day["month"] = station_day["date"].dt.month
station_day["season"] = station_day["month"].apply(assign_season)

station_day = station_day[
    (station_day["year"] >= YEAR_START) &
    (station_day["year"] <= YEAR_END)
].copy()

# ============================================================
# COMPUTE SEASONAL MEAN AQI PER STATION
# ============================================================

aqi_seasonal = (
    station_day
    .groupby(["stationid", "year", "season"], as_index=False)["aqi"]
    .mean()
    .rename(columns={"aqi": "aqi_mean"})
)

# ============================================================
# CLEAN STATION LOCATION DATA
# ============================================================

station_meta = station_meta[["stationid", "latitude", "longitude"]].copy()
station_meta["latitude"] = pd.to_numeric(station_meta["latitude"], errors="coerce")
station_meta["longitude"] = pd.to_numeric(station_meta["longitude"], errors="coerce")
station_meta = station_meta.dropna(subset=["latitude", "longitude"]).copy()

station_meta = station_meta[
    (station_meta["latitude"].between(-90, 90)) &
    (station_meta["longitude"].between(-180, 180))
].copy()

aqi_seasonal = aqi_seasonal.merge(
    station_meta,
    on="stationid",
    how="left"
)

aqi_seasonal = aqi_seasonal.dropna(subset=["latitude", "longitude"]).copy()

# ============================================================
# CLEAN FARM DATA
# ============================================================

farm_data["latitude"] = pd.to_numeric(farm_data["latitude"], errors="coerce")
farm_data["longitude"] = pd.to_numeric(farm_data["longitude"], errors="coerce")

farm_data = farm_data.dropna(subset=["latitude", "longitude", "country"]).copy()

farm_data = farm_data[
    (farm_data["latitude"].between(-90, 90)) &
    (farm_data["longitude"].between(-180, 180))
].copy()

# India only
farm_data["country"] = farm_data["country"].astype(str).str.strip().str.lower()
farm_data = farm_data[farm_data["country"] == "india"].copy()

if len(farm_data) == 0:
    raise ValueError("No valid Indian farm locations found after cleaning/filtering.")

# Optional descriptive field: crop count
if "crop" in farm_data.columns:
    farm_data["crop_count"] = farm_data["crop"].apply(infer_crop_count)
else:
    farm_data["crop_count"] = np.nan

# Detect true size column if present
farm_size_col = detect_farm_size_column(farm_data)

if farm_size_col is not None:
    farm_data[farm_size_col] = pd.to_numeric(farm_data[farm_size_col], errors="coerce")
    farm_data["farm_size_value"] = farm_data[farm_size_col]
    farm_data.loc[farm_data["farm_size_value"] < 0, "farm_size_value"] = np.nan
else:
    farm_data["farm_size_value"] = np.nan

# Keep needed fields
keep_cols = ["latitude", "longitude", "farm_size_value", "crop_count"]
if "formatted address" in farm_data.columns:
    keep_cols.append("formatted address")
elif "address" in farm_data.columns:
    keep_cols.append("address")

farm_static = farm_data[keep_cols].copy()

# Create farm label
if "formatted address" in farm_static.columns:
    farm_static["farm_label"] = farm_static["formatted address"]
elif "address" in farm_static.columns:
    farm_static["farm_label"] = farm_static["address"]
else:
    farm_static["farm_label"] = "Farm"

farm_static = farm_static.reset_index(drop=True)

# ============================================================
# COMPUTE STATIC FARM METRICS PER STATION
# ============================================================

unique_stations = (
    aqi_seasonal[["stationid", "latitude", "longitude"]]
    .drop_duplicates()
    .reset_index(drop=True)
)

station_lats = unique_stations["latitude"].to_numpy(dtype=float)
station_lons = unique_stations["longitude"].to_numpy(dtype=float)

farm_lats = farm_static["latitude"].to_numpy(dtype=float)
farm_lons = farm_static["longitude"].to_numpy(dtype=float)

distance_matrix = haversine_matrix(
    station_lats,
    station_lons,
    farm_lats,
    farm_lons
)

sorted_idx = np.argsort(distance_matrix, axis=1)
sorted_distances = np.sort(distance_matrix, axis=1)

k_used = min(K_NEAREST, sorted_distances.shape[1])

closest_distance = sorted_distances[:, 0]
mean_distance_k = sorted_distances[:, :k_used].mean(axis=1)

closest_farm_labels = farm_static["farm_label"].to_numpy()[sorted_idx[:, 0]]

farm_density_100km = (distance_matrix <= DENSITY_RADIUS_KM).sum(axis=1)

inverse_distance_score_100km = np.where(
    distance_matrix <= DENSITY_RADIUS_KM,
    1.0 / (distance_matrix + 1.0),
    0.0
).sum(axis=1)

# Crop-diversity-weighted local influence (descriptive only)
crop_counts = farm_static["crop_count"].fillna(0).to_numpy(dtype=float)
crop_weighted_inverse_score_100km = np.where(
    distance_matrix <= DENSITY_RADIUS_KM,
    crop_counts[None, :] / (distance_matrix + 1.0),
    0.0
).sum(axis=1)

# True size-weighted local influence, only if a size column exists
size_values = farm_static["farm_size_value"].fillna(0).to_numpy(dtype=float)
size_weighted_inverse_score_100km = np.where(
    distance_matrix <= DENSITY_RADIUS_KM,
    size_values[None, :] / (distance_matrix + 1.0),
    0.0
).sum(axis=1)

local_size_total_100km = np.where(
    distance_matrix <= DENSITY_RADIUS_KM,
    size_values[None, :],
    0.0
).sum(axis=1)

station_metrics = pd.DataFrame({
    "stationid": unique_stations["stationid"].values,
    "closest_farm_distance": closest_distance,
    "mean_distance_3_nearest_farms": mean_distance_k,
    "closest_farm_label": closest_farm_labels,
    "farm_density_100km": farm_density_100km,
    "inverse_distance_score_100km": inverse_distance_score_100km,
    "crop_weighted_inverse_score_100km": crop_weighted_inverse_score_100km,
    "size_weighted_inverse_score_100km": size_weighted_inverse_score_100km,
    "local_size_total_100km": local_size_total_100km,
    "log_closest_farm_distance": np.log(closest_distance + 1.0),
    "log_mean_distance_3_nearest_farms": np.log(mean_distance_k + 1.0)
})

# Merge static farm metrics into every seasonal AQI row
analysis_df = aqi_seasonal.merge(
    station_metrics,
    on="stationid",
    how="left"
)

# ============================================================
# PREPARE RESULTS
# ============================================================

results = []

add_header(results, "FARM LOCATIONS VS AQI ANALYSIS")
results.append(f"Years used for AQI: {YEAR_START} to {YEAR_END}")
results.append("AQI is segmented by season: 1=Jan-Mar, 2=Apr-Jun, 3=Jul-Sep, 4=Oct-Dec")
results.append("Farms are treated as a static geographic layer.")
results.append(f"Farm density radius: {DENSITY_RADIUS_KM} km")
results.append(f"Nearest farms used: {k_used}")
results.append(f"Unique stations used: {analysis_df['stationid'].nunique()}")
results.append(f"Total station-season-year observations: {len(analysis_df)}")
results.append(f"Total farm locations used: {len(farm_static)}")

if farm_size_col is not None:
    results.append(f"Detected farm size column: {farm_size_col}")
    results.append("Size-weighted farm influence metrics were included.")
else:
    results.append("No explicit farm size column was detected.")
    results.append("True size-weighted influence could not be computed from the provided fields.")
    results.append("Equal-weight distance and density metrics were used instead.")
    results.append("Crop-weighted influence was included only as a descriptive supplementary metric, not a true size proxy.")

# ============================================================
# OVERALL CORRELATION ANALYSIS
# ============================================================

add_header(results, "OVERALL CORRELATION ANALYSIS")

main_vars = [
    "log_closest_farm_distance",
    "log_mean_distance_3_nearest_farms",
    "farm_density_100km",
    "inverse_distance_score_100km"
]

# Add size metrics only if a size column truly exists
if farm_size_col is not None:
    main_vars.extend([
        "local_size_total_100km",
        "size_weighted_inverse_score_100km"
    ])

# Keep crop-weighted as supplementary/descriptive
main_vars.append("crop_weighted_inverse_score_100km")

for var in main_vars:
    add_corr_results(
        results,
        analysis_df,
        x_col=var,
        y_col="aqi_mean",
        label=f"AQI_mean vs {var}"
    )

# ============================================================
# SEASON-BY-SEASON CORRELATION ANALYSIS
# ============================================================

add_header(results, "SEASON-BY-SEASON CORRELATION ANALYSIS")

for season_num in [1, 2, 3, 4]:
    season_df = analysis_df[analysis_df["season"] == season_num].copy()

    results.append(f"\nSeason {season_num}")
    results.append("-" * 70)
    results.append(f"Observations: {len(season_df)}")
    results.append(f"Unique stations: {season_df['stationid'].nunique()}")

    for var in main_vars:
        add_corr_results(
            results,
            season_df,
            x_col=var,
            y_col="aqi_mean",
            label=f"Season {season_num}: AQI_mean vs {var}"
        )

# ============================================================
# REGRESSION ANALYSIS
# ============================================================

add_header(results, "OVERALL REGRESSION WITHOUT SEASON FIXED EFFECTS")

predictors = [
    "log_closest_farm_distance",
    "log_mean_distance_3_nearest_farms",
    "farm_density_100km",
    "inverse_distance_score_100km"
]

if farm_size_col is not None:
    predictors.extend([
        "local_size_total_100km",
        "size_weighted_inverse_score_100km"
    ])

results.append(
    run_hc3_regression(
        analysis_df,
        response_col="aqi_mean",
        predictor_cols=predictors,
        include_season_dummies=False
    )
)

add_header(results, "OVERALL REGRESSION WITH SEASON FIXED EFFECTS")

results.append(
    run_hc3_regression(
        analysis_df,
        response_col="aqi_mean",
        predictor_cols=predictors,
        include_season_dummies=True
    )
)

# ============================================================
# SEASON-SPECIFIC REGRESSIONS
# ============================================================

add_header(results, "SEASON-SPECIFIC REGRESSIONS")

for season_num in [1, 2, 3, 4]:
    season_df = analysis_df[analysis_df["season"] == season_num].copy()

    results.append(f"\nSeason {season_num}")
    results.append("-" * 70)
    results.append(
        run_hc3_regression(
            season_df,
            response_col="aqi_mean",
            predictor_cols=predictors,
            include_season_dummies=False
        )
    )

# ============================================================
# SAVE RESULTS
# ============================================================

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

results_path = os.path.join(OUTPUT_FOLDER, OUTPUT_TEXT_FILE)
dataset_path = os.path.join(OUTPUT_FOLDER, OUTPUT_DATA_FILE)

with open(results_path, "w", encoding="utf-8") as f:
    for line in results:
        f.write(str(line) + "\n")

analysis_df.to_csv(dataset_path, index=False)

print("Analysis complete.")
print("Results saved to:", results_path)
print("Analysis dataset saved to:", dataset_path)