import pandas as pd
import math
import requests

stations = pd.read_csv("data/stations.csv")

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
USER_AGENT = "SustainableTransportTool/1.0"

def get_coordinates(place_name):
    params = {"q": place_name, "format": "json", "limit": 1}
    try:
        response = requests.get(NOMINATIM_URL, params=params, headers={"User-Agent": USER_AGENT}, timeout=10)
        response.raise_for_status()
        results = response.json()
        if not results:
            return None
        return float(results[0]["lat"]), float(results[0]["lon"])
    except (requests.RequestException, ValueError, KeyError):
        return None

def get_place_name(station_name):
    for i in range(len(station_name)-1, 0, -1):
        if station_name[i] == "-":
            index = i
            break
    new_station_name = station_name[0:index]
    return new_station_name

stations["StationName"] = stations["StationName"].apply(get_place_name)
stations["Location"] = stations["StationName"].apply(get_coordinates)

stations["Latitude"] = stations["Location"].apply(lambda x: x[0] if x else None)
stations["Longitude"] = stations["Location"].apply(lambda x: x[1] if x else None)

stations.to_csv("data/stations_fixed.csv")