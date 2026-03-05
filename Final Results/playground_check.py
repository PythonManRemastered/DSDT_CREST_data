import pandas as pd

# Define column structure
columns = [
    'Station ID',
    'AQI mean',
    'AQI median',
    'Station longitude',
    'Station latitude'
]
tables = {}

for season in ['A', 'B', 'C', 'D']:
    for year in [2015, 2016, 2017, 2018, 2019, 2020]:
        
        table_name = f"Season_{season}_{year}"
        tables[table_name] = pd.DataFrame(columns=columns)


globals().update(tables)