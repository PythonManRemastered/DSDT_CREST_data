# -----------------------------------------------------------
# STEP 0: Import Required Libraries
# -----------------------------------------------------------

import pandas as pd
import numpy as np


# -----------------------------------------------------------
# STEP 1: Load Data
# -----------------------------------------------------------

# Load daily station AQI dataset
station_day = pd.read_csv("data/station_day.csv")

# Load station metadata (contains latitude & longitude)
station_meta = pd.read_csv("data/dataset_stationID.csv")


# -----------------------------------------------------------
# STEP 2: Keep Only Required Columns
# -----------------------------------------------------------

# From station_day, we only need StationId, Date, AQI
station_day = station_day[["StationId", "Date", "AQI"]]

# Remove rows where AQI is missing
station_day = station_day.dropna(subset=["AQI"])

# Convert AQI to numeric (in case some are stored as strings)
station_day["AQI"] = pd.to_numeric(station_day["AQI"], errors="coerce")

# Convert Date column to datetime format
station_day["Date"] = pd.to_datetime(station_day["Date"])


# -----------------------------------------------------------
# STEP 3: Extract Year and Month
# -----------------------------------------------------------

station_day["Year"] = station_day["Date"].dt.year
station_day["Month"] = station_day["Date"].dt.month


# -----------------------------------------------------------
# STEP 4: Define Seasons
# Season 1 → Months 1,2,3
# Season 2 → Months 4,5,6
# Season 3 → Months 7,8,9
# Season 4 → Months 10,11,12
# -----------------------------------------------------------

def assign_season(month):
    if month in [1, 2, 3]:
        return 1
    elif month in [4, 5, 6]:
        return 2
    elif month in [7, 8, 9]:
        return 3
    else:
        return 4

station_day["Season"] = station_day["Month"].apply(assign_season)


# -----------------------------------------------------------
# STEP 5: Compute Mean and Median AQI
# Group by StationId, Year, Season
# -----------------------------------------------------------

seasonal_stats = (
    station_day
    .groupby(["StationId", "Year", "Season"])["AQI"]
    .agg(["mean", "median"])
    .reset_index()
)

# Rename columns for clarity
seasonal_stats.rename(columns={
    "mean": "AQI_mean",
    "median": "AQI_median"
}, inplace=True)


# -----------------------------------------------------------
# STEP 6: Merge Latitude and Longitude
# -----------------------------------------------------------

# Keep only necessary columns from metadata
station_meta = station_meta[["StationId", "latitude", "longitude"]]

# Merge coordinates into seasonal statistics
seasonal_stats = seasonal_stats.merge(
    station_meta,
    on="StationId",
    how="left"
)


# -----------------------------------------------------------
# STEP 7: Create Separate Pandas Tables
# Format: Season_X_Year
# -----------------------------------------------------------

tables = {}

# Columns to keep in final seasonal tables
final_columns = [
    "StationId",
    "AQI_mean",
    "AQI_median",
    "latitude",
    "longitude"
]

# Loop through all seasons and years
for season in [1, 2, 3, 4]:
    for year in seasonal_stats["Year"].unique():
        
        table_name = f"Season_{season}_{year}"
        
        # Filter for that specific season and year
        filtered_table = seasonal_stats[
            (seasonal_stats["Season"] == season) &
            (seasonal_stats["Year"] == year)
        ][final_columns]
        
        # Reset index for clean table
        tables[table_name] = filtered_table.reset_index(drop=True)

# Optional: Make tables directly accessible by name
globals().update(tables)


# -----------------------------------------------------------
# STEP 8: Example Check
# -----------------------------------------------------------

# Example: View one seasonal table
for table_name in tables:
  tables[table_name].to_csv(f'test_1/{table_name}.csv')

    