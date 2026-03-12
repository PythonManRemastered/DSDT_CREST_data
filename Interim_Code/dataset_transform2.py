import pandas as pd
import pandas as pd

stations = pd.read_csv("data/dataset_stationID.csv")

stations[["latitude", "longitude"]] = (
    stations["Location"]
    .str.strip("()")              
    .str.split(",", expand=True)  
    .astype(float)                
)
stations.to_csv("data/dataset_stationID.csv", index=False)
print(stations)

