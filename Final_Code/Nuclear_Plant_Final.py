# ============================================================
# NUCLEAR POWER PLANT PROXIMITY VS AQI ANALYSIS
# Rigorous season-aware version
# ============================================================

# ------------------------------------------------------------
# WHERE TO ADD PATHS
# ------------------------------------------------------------
# Replace these with full paths if your files are not in the
# same folder as this Python script.
#
# Example:
# station_day_path = "/Users/yourname/Desktop/DSDT_CREST/station_day.csv"
# ------------------------------------------------------------

station_day_path = "Data/station_day.csv"
station_meta_path = "Data/dataset_stationID.csv"
nuclear_data_path = "Data/nuclear_power_plants.csv"

# ============================================================
# IMPORTS
# ============================================================

import os
import math
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

# ============================================================
# PARAMETERS
# ============================================================

YEAR_START = 2018
YEAR_END = 2020

# Radius used for local nuclear density
DENSITY_RADIUS_KM = 100

# Number of nearest plants to use for "nearest few"
K_NEAREST = 3

# Output files
OUTPUT_FOLDER = "Final_Results"
OUTPUT_TEXT_FILE = "Results_Nuclear.txt"
OUTPUT_DATA_FILE = "Nuclear_Analysis_Dataset.csv"

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_columns(df):
    """
    Standardise column names by stripping spaces and lowercasing.
    """
    df.columns = [str(col).strip().lower() for col in df.columns]
    return df


def validate_required_columns(df, required_cols, df_name):
    """
    Raise a clear error if expected columns are missing.
    """
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise KeyError(
            f"{df_name} is missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )


def assign_season(month):
    """
    Season mapping:
    1 = Jan, Feb, Mar
    2 = Apr, May, Jun
    3 = Jul, Aug, Sep
    4 = Oct, Nov, Dec
    """
    return ((month - 1) // 3) + 1


def haversine_matrix(station_lats, station_lons, plant_lats, plant_lons):
    """
    Compute a full distance matrix using the haversine formula.

    Output shape:
        (number_of_stations, number_of_plants)

    Distances are in km.
    """
    R = 6371.0

    station_lats_rad = np.radians(station_lats)[:, None]
    station_lons_rad = np.radians(station_lons)[:, None]
    plant_lats_rad = np.radians(plant_lats)[None, :]
    plant_lons_rad = np.radians(plant_lons)[None, :]

    dlat = plant_lats_rad - station_lats_rad
    dlon = plant_lons_rad - station_lons_rad

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(station_lats_rad) * np.cos(plant_lats_rad) * np.sin(dlon / 2.0) ** 2
    )

    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return R * c


def safe_pearson(x, y):
    """
    Pearson correlation with safety checks.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 3:
        return np.nan, np.nan
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan, np.nan

    return stats.pearsonr(x, y)


def safe_spearman(x, y):
    """
    Spearman correlation with safety checks.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if len(x) < 3:
        return np.nan, np.nan
    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan, np.nan

    return stats.spearmanr(x, y)


def pearson_t_test_from_r(r, n):
    """
    Formal t-test for Pearson correlation coefficient.
    """
    if pd.isna(r) or n < 3 or abs(r) >= 1:
        return np.nan, np.nan

    t_stat = (r * np.sqrt(n - 2)) / np.sqrt(1 - r**2)
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n - 2))
    return t_stat, p_value


def add_header(results, title):
    results.append("\n" + "=" * 78)
    results.append(title)
    results.append("=" * 78)


def add_corr_results(results, df, x_col, y_col, label):
    """
    Add correlation and hypothesis testing results for one variable.
    """
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
    results.append(f"Observations = {n}")
    results.append(f"Pearson r = {pearson_r}")
    results.append(f"Pearson p-value = {pearson_p}")
    results.append(f"Pearson r^2 = {pearson_r**2 if pd.notna(pearson_r) else np.nan}")
    results.append(f"Spearman rho = {spearman_rho}")
    results.append(f"Spearman p-value = {spearman_p}")
    results.append(f"Formal t-statistic = {t_stat}")
    results.append(f"Formal t-test p-value = {t_p}")


def run_hc3_regression(df, response_col, predictor_cols, include_season_dummies=False):
    """
    Run OLS regression with HC3 robust standard errors.
    Optionally include season dummies for fair season adjustment.
    """
    try:
        work_df = df.copy()

        cols_needed = [response_col] + predictor_cols
        work_df = work_df[cols_needed + (["season"] if include_season_dummies else [])].copy()

        if include_season_dummies:
            season_dummies = pd.get_dummies(work_df["season"], prefix="season", drop_first=True)
            work_df = pd.concat([work_df.drop(columns=["season"]), season_dummies], axis=1)

        work_df = work_df.dropna()

        if len(work_df) < 8:
            return "Not enough valid observations for regression."

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


# ============================================================
# LOAD DATA
# ============================================================

station_day = pd.read_csv(station_day_path)
station_meta = pd.read_csv(station_meta_path)
nuclear_data = pd.read_csv(nuclear_data_path)

# ============================================================
# CLEAN COLUMN NAMES
# ============================================================

station_day = clean_columns(station_day)
station_meta = clean_columns(station_meta)
nuclear_data = clean_columns(nuclear_data)

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
    nuclear_data,
    ["country code", "name of powerplant", "latitude", "longitude", "primary_fuel"],
    "nuclear power plant dataset"
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

# Restrict to selected years
station_day = station_day[
    (station_day["year"] >= YEAR_START) &
    (station_day["year"] <= YEAR_END)
].copy()

# ============================================================
# COMPUTE SEASONAL MEAN AQI PER STATION
# ============================================================

seasonal_stats = (
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

# Merge station coordinates
seasonal_stats = seasonal_stats.merge(
    station_meta,
    on="stationid",
    how="left"
)

seasonal_stats = seasonal_stats.dropna(subset=["latitude", "longitude"]).copy()

# ============================================================
# CLEAN NUCLEAR DATA
# ============================================================

# India only
nuclear_data["country code"] = nuclear_data["country code"].astype(str).str.strip().str.upper()
nuclear_data = nuclear_data[nuclear_data["country code"] == "IND"].copy()

# Nuclear only
nuclear_data["primary_fuel"] = nuclear_data["primary_fuel"].astype(str).str.strip().str.lower()
nuclear_data = nuclear_data[nuclear_data["primary_fuel"] == "nuclear"].copy()

# Keep needed columns
nuclear_data = nuclear_data[
    ["name of powerplant", "latitude", "longitude"]
].copy()

nuclear_data["latitude"] = pd.to_numeric(nuclear_data["latitude"], errors="coerce")
nuclear_data["longitude"] = pd.to_numeric(nuclear_data["longitude"], errors="coerce")
nuclear_data = nuclear_data.dropna(subset=["latitude", "longitude"]).copy()

nuclear_data = nuclear_data[
    (nuclear_data["latitude"].between(-90, 90)) &
    (nuclear_data["longitude"].between(-180, 180))
].copy()

if len(nuclear_data) == 0:
    raise ValueError(
        "No Indian nuclear plants were found after filtering. "
        "Check the dataset values in 'country code' and 'primary_fuel'."
    )

# ============================================================
# COMPUTE STATION-LEVEL NUCLEAR SPATIAL METRICS
# ============================================================

# We compute these once per station because plant locations do not
# change by season in your setup.

unique_stations = (
    seasonal_stats[["stationid", "latitude", "longitude"]]
    .drop_duplicates()
    .reset_index(drop=True)
)

station_lats = unique_stations["latitude"].to_numpy(dtype=float)
station_lons = unique_stations["longitude"].to_numpy(dtype=float)

plant_lats = nuclear_data["latitude"].to_numpy(dtype=float)
plant_lons = nuclear_data["longitude"].to_numpy(dtype=float)
plant_names = nuclear_data["name of powerplant"].to_numpy()

distance_matrix = haversine_matrix(
    station_lats,
    station_lons,
    plant_lats,
    plant_lons
)

# Sort each row to get nearest few plant distances
sorted_distances = np.sort(distance_matrix, axis=1)

# Closest nuclear plant
closest_nuclear_distance = sorted_distances[:, 0]

# Mean distance to the nearest few plants
# If there are fewer than K_NEAREST plants, use all available plants
k_used = min(K_NEAREST, sorted_distances.shape[1])
mean_distance_3_nearest = sorted_distances[:, :k_used].mean(axis=1)

# Closest nuclear plant name
closest_indices = np.argmin(distance_matrix, axis=1)
closest_nuclear_plant = plant_names[closest_indices]

# Density within radius
nuclear_density_100km = (distance_matrix <= DENSITY_RADIUS_KM).sum(axis=1)

# Inverse-distance score within radius
inverse_distance_score_100km = np.where(
    distance_matrix <= DENSITY_RADIUS_KM,
    1.0 / (distance_matrix + 1.0),
    0.0
).sum(axis=1)

station_metrics = pd.DataFrame({
    "stationid": unique_stations["stationid"].values,
    "closest_nuclear_distance": closest_nuclear_distance,
    "mean_distance_3_nearest": mean_distance_3_nearest,
    "closest_nuclear_plant": closest_nuclear_plant,
    "nuclear_density_100km": nuclear_density_100km,
    "inverse_distance_score_100km": inverse_distance_score_100km,
    "log_closest_nuclear_distance": np.log(closest_nuclear_distance + 1.0),
    "log_mean_distance_3_nearest": np.log(mean_distance_3_nearest + 1.0)
})

# Merge these metrics back into the seasonal AQI dataset
seasonal_stats = seasonal_stats.merge(
    station_metrics,
    on="stationid",
    how="left"
)

# ============================================================
# PREPARE RESULTS
# ============================================================

results = []

add_header(results, "NUCLEAR POWER PLANT DISTANCE VS AQI ANALYSIS")
results.append(f"Years used: {YEAR_START} to {YEAR_END}")
results.append("Season definition: 1=Jan-Mar, 2=Apr-Jun, 3=Jul-Sep, 4=Oct-Dec")
results.append(f"Nuclear density radius: {DENSITY_RADIUS_KM} km")
results.append(f"Nearest-few nuclear plants used: {k_used}")
results.append(f"Unique stations used: {seasonal_stats['stationid'].nunique()}")
results.append(f"Total station-season-year observations: {len(seasonal_stats)}")
results.append(f"Indian nuclear plants used: {len(nuclear_data)}")

# ============================================================
# OVERALL CORRELATION ANALYSIS
# ============================================================

add_header(results, "OVERALL CORRELATION ANALYSIS")

main_vars = [
    "log_closest_nuclear_distance",
    "log_mean_distance_3_nearest",
    "nuclear_density_100km",
    "inverse_distance_score_100km"
]

for var in main_vars:
    add_corr_results(
        results,
        seasonal_stats,
        x_col=var,
        y_col="aqi_mean",
        label=f"AQI_mean vs {var}"
    )

# ============================================================
# SEASON-BY-SEASON ANALYSIS
# ============================================================

add_header(results, "SEASON-BY-SEASON CORRELATION ANALYSIS")

for season_num in [1, 2, 3, 4]:
    season_df = seasonal_stats[seasonal_stats["season"] == season_num].copy()

    results.append(f"\nSeason {season_num}")
    results.append("-" * 60)
    results.append(f"Observations = {len(season_df)}")
    results.append(f"Unique stations = {season_df['stationid'].nunique()}")

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

basic_predictors = [
    "log_closest_nuclear_distance",
    "log_mean_distance_3_nearest",
    "nuclear_density_100km",
    "inverse_distance_score_100km"
]

results.append(
    run_hc3_regression(
        seasonal_stats,
        response_col="aqi_mean",
        predictor_cols=basic_predictors,
        include_season_dummies=False
    )
)

add_header(results, "OVERALL REGRESSION WITH SEASON FIXED EFFECTS")

results.append(
    run_hc3_regression(
        seasonal_stats,
        response_col="aqi_mean",
        predictor_cols=basic_predictors,
        include_season_dummies=True
    )
)

# ============================================================
# SEASON-SPECIFIC REGRESSIONS
# ============================================================

add_header(results, "SEASON-SPECIFIC REGRESSIONS")

for season_num in [1, 2, 3, 4]:
    season_df = seasonal_stats[seasonal_stats["season"] == season_num].copy()

    results.append(f"\nSeason {season_num}")
    results.append("-" * 60)
    results.append(
        run_hc3_regression(
            season_df,
            response_col="aqi_mean",
            predictor_cols=basic_predictors,
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

seasonal_stats.to_csv(dataset_path, index=False)

print("Analysis complete.")
print("Results saved to:", results_path)
print("Analysis dataset saved to:", dataset_path)