import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

stations = pd.read_csv("data/dataset_stationID.csv")

xpoints = np.array(stations["longitude"])
ypoints = np.array(stations["latitude"])

plt.scatter(xpoints, ypoints, marker='+')

plt.xlabel("Longitude")
plt.ylabel("Latitude")

plt.show()
