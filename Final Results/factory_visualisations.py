import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

DATA_FILE = "data/67e185e8ed1dc5f64781b97b_Industrial_and_Manufacturing_and_Resources_sample_data.csv"

locations = pd.read_csv(DATA_FILE)

coords = locations[["longitude", "latitude"]].copy()
coords["longitude"] = pd.to_numeric(coords["longitude"], errors="coerce")
coords["latitude"] = pd.to_numeric(coords["latitude"], errors="coerce")
coords = coords.dropna()

fig, ax = plt.subplots(figsize=(8, 8))
img = mpimg.imread("data/India-Outline-Map-Colourful-1-882x1024.png")

min_lon = coords["longitude"].min()
max_lon = coords["longitude"].max()
min_lat = coords["latitude"].min()
max_lat = coords["latitude"].max()

lon_span = max_lon - min_lon
lat_span = max_lat - min_lat

img_scale = 1.03
lon_center = (min_lon + max_lon) / 2
lat_center = (min_lat + max_lat) / 2
img_half_lon = (lon_span / 2) * img_scale
img_half_lat = (lat_span / 2) * img_scale

img_extent = [
    lon_center - img_half_lon,
    lon_center + img_half_lon,
    lat_center - img_half_lat,
    lat_center + img_half_lat,
]

img_rgb = img[..., :3]
if img_rgb.dtype != np.float32 and img_rgb.max() > 1.0:
    img_rgb = img_rgb / 255.0
non_white = ~(img_rgb > 0.92).all(axis=2)
img_height, img_width = non_white.shape

xs = (coords["longitude"] - img_extent[0]) / (img_extent[1] - img_extent[0]) * (img_width - 1)
ys = (img_extent[3] - coords["latitude"]) / (img_extent[3] - img_extent[2]) * (img_height - 1)
xs = xs.round().astype(int).clip(0, img_width - 1)
ys = ys.round().astype(int).clip(0, img_height - 1)
on_land = non_white[ys, xs]
coords = coords[on_land]

ax.imshow(
    img,
    extent=img_extent,
    origin="upper",
)

ax.scatter(coords["longitude"], coords["latitude"], s=12, alpha=0.7)
ax.set_title("Factory Locations (Longitude vs Latitude)")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
ax.grid(True, linestyle="--", alpha=0.4)
zoom_pad = 0.03
ax.set_xlim(min_lon - lon_span * zoom_pad, max_lon + lon_span * zoom_pad)
ax.set_ylim(min_lat - lat_span * zoom_pad, max_lat + lat_span * zoom_pad)
plt.tight_layout()
plt.show()
