# ============================================================
# RAINFALL VS AQI ANALYSIS
# AQI segmented by season, rainfall matched by city/district
# and state, with configurable city/state weighting
# ============================================================

# ------------------------------------------------------------
# WHERE TO ADD PATHS
# ------------------------------------------------------------
# Replace these if needed.
# Example:
# station_day_path = "/Users/yourname/Desktop/DSDT_CREST/station_day.csv"
# ------------------------------------------------------------

station_day_path = "Data/station_day.csv"
station_meta_path = "Data/dataset_stationID.csv"
rainfall_data_path = "Data/rain_dataset.csv"

# ============================================================
# IMPORTS
# ============================================================

import os
import re
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

# ============================================================
# PARAMETERS
# ============================================================

YEAR_START = 2018
YEAR_END = 2020

OUTPUT_FOLDER = "Final_Results"
OUTPUT_TEXT_FILE = "Results_Rainfall.txt"
OUTPUT_DATA_FILE = "Rainfall_Analysis_Dataset.csv"

# ------------------------------------------------------------
# CITY/STATE WEIGHTING SETTINGS
# ------------------------------------------------------------
# SIMPLE WAY TO CHANGE THE RATIO:
# If both city/district rainfall and state rainfall are available,
# the weighted rainfall is:
#
#   weighted_rain = (CITY_WEIGHT * city_rain + STATE_WEIGHT * state_rain)
#                   / (CITY_WEIGHT + STATE_WEIGHT)
#
# To make city rainfall count even more, increase CITY_WEIGHT.
# Example:
#   CITY_WEIGHT = 10
#   STATE_WEIGHT = 1
#
# If you want to use ONLY city when city exists, set:
#   USE_BOTH_CITY_AND_STATE = False
#
# If city rainfall is missing, the code automatically falls back to state.
# ------------------------------------------------------------

USE_BOTH_CITY_AND_STATE = True
CITY_WEIGHT = 10.0
STATE_WEIGHT = 1.0

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_columns(df):
    """
    Standardise column names:
    - convert to string
    - strip whitespace
    - remove quote marks
    - lowercase
    """
    cleaned = []
    for col in df.columns:
        c = str(col).strip().replace('"', "").replace("'", "").lower()
        cleaned.append(c)
    df.columns = cleaned
    return df


def clean_text_value(x):
    """
    Standardise text fields for matching.
    """
    if pd.isna(x):
        return np.nan
    x = str(x).strip().replace('"', "").replace("'", "").lower()
    x = re.sub(r"\s+", " ", x)
    return x


def validate_required_columns(df, required_cols, df_name):
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise KeyError(
            f"{df_name} is missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )


def assign_season(month):
    """
    Season 1 = Jan-Mar
    Season 2 = Apr-Jun
    Season 3 = Jul-Sep
    Season 4 = Oct-Dec
    """
    return ((month - 1) // 3) + 1


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
    results.append("\n" + "=" * 94)
    results.append(title)
    results.append("=" * 94)


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


def weighted_city_state_rain(city_rain, state_rain):
    """
    Apply the user-defined city/state weighting rule.
    """
    city_exists = pd.notna(city_rain)
    state_exists = pd.notna(state_rain)

    if city_exists and state_exists:
        if USE_BOTH_CITY_AND_STATE:
            return (CITY_WEIGHT * city_rain + STATE_WEIGHT * state_rain) / (CITY_WEIGHT + STATE_WEIGHT)
        else:
            return city_rain

    if city_exists:
        return city_rain

    if state_exists:
        return state_rain

    return np.nan


# ============================================================
# LOAD DATA
# ============================================================

station_day = pd.read_csv(station_day_path)
station_meta = pd.read_csv(station_meta_path)

# Rainfall file is semicolon-separated in your example
rainfall_data = pd.read_csv(rainfall_data_path, sep=";", engine="python")

# ============================================================
# CLEAN COLUMN NAMES
# ============================================================

station_day = clean_columns(station_day)
station_meta = clean_columns(station_meta)
rainfall_data = clean_columns(rainfall_data)

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
    ["stationid", "city", "state"],
    "dataset_stationID"
)

validate_required_columns(
    rainfall_data,
    ["state", "district", "month"],
    "rainfall dataset"
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
# CLEAN STATION METADATA
# ============================================================

station_meta = station_meta[["stationid", "city", "state"]].copy()
station_meta["city_clean"] = station_meta["city"].apply(clean_text_value)
station_meta["state_clean"] = station_meta["state"].apply(clean_text_value)

aqi_seasonal = aqi_seasonal.merge(
    station_meta[["stationid", "city_clean", "state_clean"]],
    on="stationid",
    how="left"
)

# ============================================================
# CLEAN RAINFALL DATA
# ============================================================

# Clean text fields
rainfall_data["state_clean"] = rainfall_data["state"].apply(clean_text_value)
rainfall_data["district_clean"] = rainfall_data["district"].apply(clean_text_value)

# Month numeric
rainfall_data["month"] = pd.to_numeric(rainfall_data["month"], errors="coerce")
rainfall_data = rainfall_data.dropna(subset=["month"]).copy()
rainfall_data["month"] = rainfall_data["month"].astype(int)
rainfall_data = rainfall_data[rainfall_data["month"].between(1, 12)].copy()

# Detect daily rainfall columns
day_columns = []
for col in rainfall_data.columns:
    if re.fullmatch(r"\d+(st|nd|rd|th)", col):
        day_columns.append(col)

if len(day_columns) == 0:
    raise ValueError(
        "No daily rainfall columns like '1st', '2nd', ... '31st' were found."
    )

# Convert daily rainfall values to numeric
for col in day_columns:
    rainfall_data[col] = pd.to_numeric(rainfall_data[col], errors="coerce")

# Mean rainfall for that district-month row
rainfall_data["row_mean_rainfall_mm"] = rainfall_data[day_columns].mean(axis=1, skipna=True)

# Season
rainfall_data["season"] = rainfall_data["month"].apply(assign_season)

# ============================================================
# BUILD SEASONAL RAINFALL TABLES
# ============================================================

# District-level seasonal rainfall
district_seasonal_rain = (
    rainfall_data
    .groupby(["state_clean", "district_clean", "season"], as_index=False)["row_mean_rainfall_mm"]
    .mean()
    .rename(columns={"row_mean_rainfall_mm": "district_seasonal_rainfall_mm"})
)

# State-level seasonal rainfall
state_seasonal_rain = (
    rainfall_data
    .groupby(["state_clean", "season"], as_index=False)["row_mean_rainfall_mm"]
    .mean()
    .rename(columns={"row_mean_rainfall_mm": "state_seasonal_rainfall_mm"})
)

# ============================================================
# MERGE RAINFALL INTO AQI DATA
# ============================================================

analysis_df = aqi_seasonal.merge(
    district_seasonal_rain,
    left_on=["state_clean", "city_clean", "season"],
    right_on=["state_clean", "district_clean", "season"],
    how="left"
)

analysis_df = analysis_df.merge(
    state_seasonal_rain,
    on=["state_clean", "season"],
    how="left"
)

# Weighted rainfall metric
analysis_df["weighted_seasonal_rainfall_mm"] = analysis_df.apply(
    lambda row: weighted_city_state_rain(
        row["district_seasonal_rainfall_mm"],
        row["state_seasonal_rainfall_mm"]
    ),
    axis=1
)

# Track which source was used
def rainfall_source_label(row):
    district_exists = pd.notna(row["district_seasonal_rainfall_mm"])
    state_exists = pd.notna(row["state_seasonal_rainfall_mm"])

    if district_exists and state_exists:
        if USE_BOTH_CITY_AND_STATE:
            return "district+state_weighted"
        return "district_only"
    if district_exists:
        return "district_only"
    if state_exists:
        return "state_only"
    return "missing"

analysis_df["rainfall_source"] = analysis_df.apply(rainfall_source_label, axis=1)

# Optional log transform
analysis_df["log_weighted_seasonal_rainfall_mm"] = np.log(analysis_df["weighted_seasonal_rainfall_mm"] + 1.0)

# ============================================================
# PREPARE RESULTS
# ============================================================

results = []

add_header(results, "RAINFALL VS AQI ANALYSIS")
results.append(f"Years used for AQI: {YEAR_START} to {YEAR_END}")
results.append("AQI is segmented by season: 1=Jan-Mar, 2=Apr-Jun, 3=Jul-Sep, 4=Oct-Dec")
results.append("Rainfall is matched by district/city first, then state as fallback.")
results.append(f"USE_BOTH_CITY_AND_STATE = {USE_BOTH_CITY_AND_STATE}")
results.append(f"CITY_WEIGHT = {CITY_WEIGHT}")
results.append(f"STATE_WEIGHT = {STATE_WEIGHT}")
results.append(f"Total station-season-year observations: {len(analysis_df)}")
results.append(f"Unique stations used: {analysis_df['stationid'].nunique()}")

source_counts = analysis_df["rainfall_source"].value_counts(dropna=False).to_dict()
results.append("Rainfall source usage:")
for k, v in source_counts.items():
    results.append(f"  {k}: {v}")

# ============================================================
# OVERALL CORRELATION ANALYSIS
# ============================================================

add_header(results, "OVERALL CORRELATION ANALYSIS")

rain_vars = [
    "district_seasonal_rainfall_mm",
    "state_seasonal_rainfall_mm",
    "weighted_seasonal_rainfall_mm",
    "log_weighted_seasonal_rainfall_mm"
]

for var in rain_vars:
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
    results.append("-" * 75)
    results.append(f"Observations: {len(season_df)}")
    results.append(f"Unique stations: {season_df['stationid'].nunique()}")

    for var in rain_vars:
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
    "weighted_seasonal_rainfall_mm"
]

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

add_header(results, "OVERALL REGRESSION WITH LOG RAINFALL AND SEASON FIXED EFFECTS")

results.append(
    run_hc3_regression(
        analysis_df,
        response_col="aqi_mean",
        predictor_cols=["log_weighted_seasonal_rainfall_mm"],
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
    results.append("-" * 75)
    results.append(
        run_hc3_regression(
            season_df,
            response_col="aqi_mean",
            predictor_cols=["weighted_seasonal_rainfall_mm"],
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