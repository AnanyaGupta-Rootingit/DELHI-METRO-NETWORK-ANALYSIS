# 📌 Install required packages (only in Colab)
!pip install geopandas folium

# 📌 Import libraries
import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import HeatMap
import matplotlib.pyplot as plt

# 📂 Load the Excel file from your drive or upload directly in Colab
from google.colab import files
uploaded = files.upload()

# Assume you uploaded 'Delhi_Metro_Master.xlsx'
file_path = 'Delhi_Metro_Master.xlsx'

# Load Stations sheet
stations_df = pd.read_excel(file_path, sheet_name='Stations')

# Load Ward Population (optional for later steps)
wards_df = pd.read_excel(file_path, sheet_name='Ward_Population')

# 📌 Inspect first few rows
print(stations_df.head())
print(wards_df.head())

# ➤ Create a base map centered in Delhi
delhi_map = folium.Map(location=[28.6139, 77.2090], zoom_start=11)

# ➤ Add stations as markers colored by Line
line_colors = {
    'Red': 'red',
    'Blue': 'blue',
    'Yellow': 'orange',
    'Green': 'green',
    'Violet': 'purple',
    'Magenta': 'pink',
    'Grey': 'gray',
    'Pink': 'lightpink',
    'Rapid Metro': 'cadetblue'
}

# Filter out rows with NaN in Latitude_network or Longitude_network
stations_df_cleaned = stations_df.dropna(subset=['Latitude_network', 'Longitude_network'])

for idx, row in stations_df_cleaned.iterrows():
    folium.CircleMarker(
        location=[row['Latitude_network'], row['Longitude_network']],
        radius=5,
        color=line_colors.get(row['Line'], 'black'),
        fill=True,
        fill_opacity=0.7,
        popup=f"{row['Station']} ({row['Line']})"
    ).add_to(delhi_map)

# ➤ Display the map
delhi_map


from folium.plugins import HeatMap

# ➤ Prepare data for heatmap (list of [lat, lon])
# Filter out rows with NaN in Latitude_network or Longitude_network
heat_data = stations_df.dropna(subset=['Latitude_network', 'Longitude_network'])[['Latitude_network', 'Longitude_network']].values.tolist()

# ➤ Add HeatMap layer
HeatMap(heat_data, radius=15, blur=10, max_zoom=13).add_to(delhi_map)

# ➤ Display the map with heatmap
delhi_map

from folium import FeatureGroup, LayerControl

# Create feature groups
stations_fg = FeatureGroup(name='Stations')
heatmap_fg = FeatureGroup(name='Heatmap')

# Add stations to the feature group
# Filter out rows with NaN in Latitude_network or Longitude_network
stations_df_cleaned = stations_df.dropna(subset=['Latitude_network', 'Longitude_network'])

for idx, row in stations_df_cleaned.iterrows():
    folium.CircleMarker(
        location=[row['Latitude_network'], row['Longitude_network']],
        radius=5,
        color=line_colors.get(row['Line'], 'black'),
        fill=True,
        fill_opacity=0.7,
        popup=f"{row['Station']} ({row['Line']})"
    ).add_to(stations_fg)

# Add heatmap to the feature group using the already prepared heat_data
HeatMap(heat_data, radius=15, blur=10, max_zoom=13).add_to(heatmap_fg)

# Add both groups to the map
stations_fg.add_to(delhi_map)
heatmap_fg.add_to(delhi_map)

# Add layer control
LayerControl().add_to(delhi_map)

# Display the map
delhi_map


import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

# ➤ Group by Line and count stations
station_counts = stations_df.groupby('Line')['Station'].count().reset_index()
station_counts = station_counts.rename(columns={'Station': 'Station Count'})

# ➤ Display the table
print(station_counts)

# ➤ Plot using matplotlib
plt.figure(figsize=(10,6))
plt.bar(station_counts['Line'], station_counts['Station Count'], color='skyblue')
plt.xticks(rotation=45, ha='right')
plt.title('Number of Stations per Metro Line')
plt.xlabel('Line')
plt.ylabel('Station Count')
plt.tight_layout()
plt.show()

# ➤ Plot using plotly for interactivity
fig = px.bar(station_counts, x='Line', y='Station Count', color='Line',
             title='Number of Stations per Metro Line',
             labels={'Line': 'Metro Line', 'Station Count': 'Stations'})
fig.update_layout(xaxis_tickangle=-45)
fig.show()

import geopandas as gpd
from shapely.geometry import Point

# ➤ Convert the stations DataFrame to a GeoDataFrame
# Filter out rows with NaN in Latitude_network or Longitude_network
stations_df_cleaned = stations_df.dropna(subset=['Latitude_network', 'Longitude_network'])

gdf = gpd.GeoDataFrame(
    stations_df_cleaned,
    geometry=gpd.points_from_xy(stations_df_cleaned['Longitude_network'], stations_df_cleaned['Latitude_network']),
    crs="EPSG:4326"  # WGS84 Latitude/Longitude
)

# ➤ Project to EPSG:3857 (meters) for accurate buffering
gdf = gdf.to_crs(epsg=3857)

# ➤ Create buffer zones (500 meters and 1000 meters)
gdf['buffer_500m'] = gdf.geometry.buffer(500)
gdf['buffer_1km'] = gdf.geometry.buffer(1000)

# ➤ Project back to EPSG:4326 for mapping
gdf = gdf.to_crs(epsg=4326)

# ➤ Create a folium map centered in Delhi
buffer_map = folium.Map(location=[28.6139, 77.2090], zoom_start=11)

# ➤ Add buffers (500m in blue, 1km in red)
for idx, row in gdf.iterrows():
    folium.GeoJson(row['buffer_500m'],
                   style_function=lambda x: {'fillColor': 'blue', 'color': 'blue', 'weight': 1, 'fillOpacity': 0.2}
                  ).add_to(buffer_map)
    folium.GeoJson(row['buffer_1km'],
                   style_function=lambda x: {'fillColor': 'red', 'color': 'red', 'weight': 1, 'fillOpacity': 0.1}
                  ).add_to(buffer_map)

# ➤ Add station markers on top
for idx, row in gdf.iterrows():
    folium.CircleMarker(
        location=[row.geometry.y, row.geometry.x],
        radius=3,
        color='black',
        fill=True,
        fill_opacity=0.7,
        popup=f"{row['Station']} ({row['Line']})"
    ).add_to(buffer_map)

# ➤ Display the map with buffers
buffer_map

from shapely.geometry import Point

# ➤ Create a GeoDataFrame for wards using approximate centroids
# Here, you can manually input coordinates or simulate random nearby points for demonstration
# For now, we create placeholder points assuming they are uniformly distributed near the city center

import numpy as np

# Generate random points around Delhi's center for demo purposes (replace with actual data if available)
np.random.seed(0)
wards_df['Latitude'] = 28.6139 + np.random.uniform(-0.1, 0.1, size=len(wards_df))
wards_df['Longitude'] = 77.2090 + np.random.uniform(-0.1, 0.1, size=len(wards_df))

# Create GeoDataFrame
wards_gdf = gpd.GeoDataFrame(
    wards_df,
    geometry=gpd.points_from_xy(wards_df['Longitude'], wards_df['Latitude']),
    crs="EPSG:4326"
)
# Convert wards to EPSG:3857 for spatial operations
wards_gdf = wards_gdf.to_crs(epsg=3857)
station_buffers = gdf.to_crs(epsg=3857)
# Initialize population sums
wards_df['covered_500m'] = 0
wards_df['covered_1km'] = 0

for idx_w, ward in wards_gdf.iterrows():
    ward_point = ward.geometry
    pop = ward['Population']

    # Check if the ward intersects any 500m buffer
    if station_buffers['buffer_500m'].intersects(ward_point).any():
        wards_df.at[idx_w, 'covered_500m'] = pop

    # Check if the ward intersects any 1km buffer
    if station_buffers['buffer_1km'].intersects(ward_point).any():
        wards_df.at[idx_w, 'covered_1km'] = pop

# Sum total population covered
total_covered_500m = wards_df['covered_500m'].sum()
total_covered_1km = wards_df['covered_1km'].sum()

print(f"Total population covered within 500m: {total_covered_500m}")
print(f"Total population covered within 1km: {total_covered_1km}")

import matplotlib.pyplot as plt

# Total population
total_population = wards_df['Population'].sum()

# Population covered within 500m and 1km
covered_500m = wards_df['covered_500m'].sum()
covered_1km = wards_df['covered_1km'].sum()

# Uncovered population
uncovered_500m = total_population - covered_500m
uncovered_1km = total_population - covered_1km

# Prepare data for plotting
coverage_data = {
    '500m': [covered_500m, uncovered_500m],
    '1km': [covered_1km, uncovered_1km]
}

# Plotting
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# 500m coverage
ax[0].bar(['Covered', 'Uncovered'], coverage_data['500m'], color=['green', 'red'])
ax[0].set_title('Population Coverage within 500m')
ax[0].set_ylabel('Population')

# 1km coverage
ax[1].bar(['Covered', 'Uncovered'], coverage_data['1km'], color=['green', 'red'])
ax[1].set_title('Population Coverage within 1km')

plt.tight_layout()
plt.show()
# Wards not covered within 500m
underserved_500m = wards_df[wards_df['covered_500m'] == 0].sort_values(by='Population', ascending=False)

# Wards not covered within 1km
underserved_1km = wards_df[wards_df['covered_1km'] == 0].sort_values(by='Population', ascending=False)

print("Top 5 underserved wards within 500m:")
print(underserved_500m[['Ward', 'Population']].head())

print("\nTop 5 underserved wards within 1km:")
print(underserved_1km[['Ward', 'Population']].head())
# Create a base map
ward_map = folium.Map(location=[28.6139, 77.2090], zoom_start=11)

# Add wards to the map, coloring by coverage within 1km
for idx, row in wards_gdf.iterrows():
    color = 'green' if wards_df.loc[idx, 'covered_1km'] > 0 else 'red'
    folium.CircleMarker(
        location=[row.geometry.y, row.geometry.x],
        radius=7,
        color=color,
        fill=True,
        fill_opacity=0.6,
        popup=f"{wards_df.loc[idx, 'Ward']} ({wards_df.loc[idx, 'Population']} people)"
    ).add_to(ward_map)

# Display the map
ward_map

import ipywidgets as widgets
from IPython.display import display, clear_output
import geopandas as gpd
from shapely.geometry import Point
import folium
import numpy as np

# Unique lines from data
unique_lines = stations_df['Line'].unique().tolist()

line_selector = widgets.SelectMultiple(
    options=unique_lines,
    value=unique_lines,
    description='Lines:',
    disabled=False
)
# Corrected column name for layout and removed NaN values
unique_layouts = stations_df['Station Layout'].dropna().unique().tolist()

layout_selector = widgets.SelectMultiple(
    options=unique_layouts,
    value=unique_layouts,
    description='Layouts:',
    disabled=False
)
buffer_selector = widgets.RadioButtons(
    options=[500, 1000],
    value=500,
    description='Buffer (m):',
    disabled=False
)
display(line_selector, layout_selector, buffer_selector)
def update_map(line_values, layout_values, buffer_distance):
    clear_output(wait=True)
    display(line_selector, layout_selector, buffer_selector)

    # Filter stations by selected lines and layouts, and drop rows with NaN in location
    filtered_stations = stations_df[
        (stations_df['Line'].isin(line_values)) &
        (stations_df['Station Layout'].isin(layout_values))
    ].dropna(subset=['Latitude_network', 'Longitude_network']) # Added dropna here

    # Convert to GeoDataFrame
    gdf_filtered = gpd.GeoDataFrame(
        filtered_stations,
        geometry=gpd.points_from_xy(filtered_stations['Longitude_network'], filtered_stations['Latitude_network']), # Corrected column names
        crs="EPSG:4326"
    ).to_crs(epsg=3857)

    # Create buffers based on selected distance
    gdf_filtered['buffer'] = gdf_filtered.geometry.buffer(buffer_distance)
    gdf_filtered = gdf_filtered.to_crs(epsg=4326)

    # Create map
    map_filtered = folium.Map(location=[28.6139, 77.2090], zoom_start=11)

    # Add buffers
    for idx, row in gdf_filtered.iterrows():
        folium.GeoJson(row['buffer'],
                       style_function=lambda x: {'fillColor': 'blue', 'color': 'blue', 'weight': 1, 'fillOpacity': 0.2}
                      ).add_to(map_filtered)

    # Add station markers
    for idx, row in gdf_filtered.iterrows():
        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=5,
            color='black',
            fill=True,
            fill_opacity=0.7,
            popup=f"{row['Station']} ({row['Line']})"
        ).add_to(map_filtered)

    # Display the map
    display(map_filtered)
widgets.interactive(
    update_map,
    line_values=line_selector,
    layout_values=layout_selector,
    buffer_distance=buffer_selector
)

import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

# ➤ Load dataset (adjust the path or upload method as needed)
file_path = 'Delhi_Metro_Master.xlsx'  # Update with actual path
ridership_df = pd.read_excel(file_path, sheet_name='Ridership_RS')

# ➤ Explore dataset
print(ridership_df.info())
print(ridership_df.head())

# ➤ Annual Ridership Trends Plot using Matplotlib
plt.figure(figsize=(10,6))
plt.plot(ridership_df['Year'], ridership_df['Ridership (Lakhs/day)'], marker='o', color='blue')
plt.title('Annual Ridership Trends (2013–2022)', fontsize=14)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Ridership (Lakhs/day)', fontsize=12)
plt.grid(True)
plt.xticks(ridership_df['Year'])
plt.tight_layout()
plt.show()

# ➤ Annual Ridership Trends Plot using Plotly
fig = px.line(ridership_df, x='Year', y='Ridership (Lakhs/day)', markers=True,
              title='Annual Ridership Trends (2013–2022)',
              labels={'Ridership (Lakhs/day)': 'Ridership', 'Year': 'Year'})
fig.update_traces(line=dict(width=3), marker=dict(size=8))
fig.show()

# ➤ COVID-19 Impact Highlighting Plot
fig = go.Figure()
fig.add_trace(go.Scatter(x=ridership_df['Year'], y=ridership_df['Ridership (Lakhs/day)'],
                         mode='lines+markers', name='Ridership'))

# ➤ Highlight the COVID period (example: 2020 and 2021)
fig.add_vrect(x0=2020, x1=2021, fillcolor="red", opacity=0.2,
              layer="below", line_width=0,
              annotation_text="COVID-19 Impact", annotation_position="top left")

fig.update_layout(title='Ridership Trend with COVID-19 Impact (2013–2022)',
                  xaxis_title='Year',
                  yaxis_title='Ridership (Lakhs/day)',
                  xaxis=dict(tickmode='linear'))
fig.show()

# ➤ Rolling Average (3-Year) to Smooth Ridership
ridership_df['Rolling_Avg'] = ridership_df['Ridership (Lakhs/day)'].rolling(window=3, center=True).mean()

plt.figure(figsize=(10,6))
plt.plot(ridership_df['Year'], ridership_df['Ridership (Lakhs/day)'], marker='o', label='Actual Ridership')
plt.plot(ridership_df['Year'], ridership_df['Rolling_Avg'], marker='x', linestyle='--', color='orange', label='3-Year Rolling Average')
plt.title('Ridership Trend with Rolling Average', fontsize=14)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Ridership (Lakhs/day)', fontsize=12)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ➤ Year-over-Year Percentage Change
ridership_df['YoY_Change'] = ridership_df['Ridership (Lakhs/day)'].pct_change() * 100

plt.figure(figsize=(10,6))
plt.bar(ridership_df['Year'], ridership_df['YoY_Change'], color='purple')
plt.title('Year-over-Year Ridership Change (%)', fontsize=14)
plt.xlabel('Year', fontsize=12)
plt.ylabel('YoY Change (%)', fontsize=12)
plt.grid(axis='y')
plt.tight_layout()
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.figure_factory as ff
import plotly.colors as pc
import re

# ➤ Load dataset
file_path = '/content/Delhi_Metro_Master (2).xlsx'
stations_df = pd.read_excel(file_path, sheet_name='Stations')
ridership_df = pd.read_excel(file_path, sheet_name='Ridership_RS')
ridership_long_df = pd.read_excel(file_path, sheet_name='Ridership_Long')

# ➤ Data Cleaning

# Clean 'Annual Ridership': remove commas and any non-digit characters
def clean_ridership(value):
    if pd.isnull(value):
        return 0
    clean_value = re.sub(r'[^\d]', '', str(value))
    return int(clean_value) if clean_value else 0

ridership_long_df['Annual Ridership'] = ridership_long_df['Annual Ridership'].apply(clean_ridership)

# Clean 'Route Length (km)'
ridership_long_df['Route Length (km)'] = pd.to_numeric(ridership_long_df['Route Length (km)'], errors='coerce')

# Parse 'Opening Date'
stations_df['Opening Date'] = pd.to_datetime(stations_df['Opening Date'], errors='coerce')

# Clean 'Distance from Start (km)'
stations_df['Distance from Start (km)'] = pd.to_numeric(stations_df['Distance from Start (km)'], errors='coerce')

# Strip extra spaces from 'Line'
stations_df['Line'] = stations_df['Line'].str.strip()
ridership_long_df['Year'] = ridership_long_df['Year'].astype(str).str.strip()

# ➤ 1. Number of Stations per Line
station_counts = stations_df.groupby('Line').size().reset_index(name='Station_Count')

plt.figure(figsize=(10,6))
plt.bar(station_counts['Line'], station_counts['Station_Count'], color='skyblue')
plt.title('Number of Stations per Line', fontsize=14)
plt.xlabel('Line', fontsize=12)
plt.ylabel('Number of Stations', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# ➤ 2. Line Length vs Ridership Efficiency
line_length = stations_df.groupby('Line')['Distance from Start (km)'].max().reset_index(name='Network_Length')
latest_year_data = ridership_long_df.iloc[-1]
total_ridership = latest_year_data['Annual Ridership']
total_length = latest_year_data['Route Length (km)']

station_counts = station_counts.merge(line_length, on='Line')
station_counts['Ridership'] = station_counts['Station_Count'] / station_counts['Station_Count'].sum() * total_ridership
station_counts['Efficiency'] = station_counts['Ridership'] / station_counts['Network_Length']

fig = px.scatter(station_counts, x='Network_Length', y='Efficiency',
                 size='Ridership', hover_name='Line',
                 title='Line Length vs Ridership Efficiency',
                 labels={'Network_Length': 'Network Length (km)', 'Efficiency': 'Ridership per km'})
fig.update_layout(xaxis=dict(showgrid=True), yaxis=dict(showgrid=True))
fig.show()

# ➤ 3. Timeline of Line Expansion (Gantt-style)

# Drop rows with missing 'Opening Date'
line_opening = stations_df.dropna(subset=['Opening Date']).groupby('Line')['Opening Date'].min().reset_index()
line_opening['Start_Year'] = line_opening['Opening Date'].dt.year
line_opening['End_Year'] = pd.Timestamp.now().year

gantt_df = line_opening[['Line', 'Start_Year', 'End_Year']]
gantt_df = gantt_df.rename(columns={'Line': 'Task', 'Start_Year': 'Start', 'End_Year': 'Finish'})

# Assign enough colors for all tasks
unique_tasks = gantt_df['Task'].nunique()
color_list = pc.qualitative.Safe * (unique_tasks // len(pc.qualitative.Safe) + 1)
color_list = color_list[:unique_tasks]

fig = ff.create_gantt(gantt_df, index_col='Task', show_colorbar=True, group_tasks=True,
                      title='Timeline of Line Expansion', colors=color_list)
fig.show()

import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

# ➤ Load dataset
file_path = '/content/Delhi_Metro_Master (2).xlsx'
stations_df = pd.read_excel(file_path, sheet_name='Stations')

# ➤ Data Cleaning

# Strip spaces from 'Station Layout' and 'Line'
stations_df['Station Layout'] = stations_df['Station Layout'].str.strip()
stations_df['Line'] = stations_df['Line'].str.strip()

# Drop rows with missing Layout information if any
stations_layout_df = stations_df.dropna(subset=['Station Layout'])

# ➤ 1. Elevated vs Underground vs At-grade distribution (overall)

layout_counts = stations_layout_df['Station Layout'].value_counts().reset_index()
layout_counts.columns = ['Layout', 'Count']

plt.figure(figsize=(8,6))
plt.pie(layout_counts['Count'], labels=layout_counts['Layout'], autopct='%1.1f%%', startangle=140)
plt.title('Overall Station Layout Distribution')
plt.tight_layout()
plt.show()

# ➤ 2. Layout preference per line

layout_line = stations_layout_df.groupby(['Line', 'Station Layout']).size().reset_index(name='Count')

fig = px.bar(layout_line, x='Line', y='Count', color='Station Layout',
             title='Layout Preference per Line',
             labels={'Count': 'Number of Stations', 'Station Layout': 'Layout'}) # Corrected label
fig.update_layout(barmode='stack', xaxis_tickangle=-45)
fig.show()

# ➤ 3. Spatial Mapping of Station Layouts

# Drop rows missing coordinates if any
stations_map_df = stations_layout_df.dropna(subset=['Latitude_network', 'Longitude_network'])

fig = px.scatter_mapbox(stations_map_df, lat='Latitude_network', lon='Longitude_network', color='Station Layout',
                        hover_name='Station', # Corrected hover_name
                        zoom=10, height=600,
                        title='Spatial Mapping of Station Layouts')

fig.update_layout(mapbox_style='open-street-map')
fig.update_layout(margin={"r":0,"t":50,"l":0,"b":0})
fig.show()

import pandas as pd
import numpy as np
from geopy.distance import geodesic
import plotly.express as px
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point

# ➤ Load dataset
file_path = '/content/Delhi_Metro_Master (2).xlsx'
stations_df = pd.read_excel(file_path, sheet_name='Stations')
wards_df = pd.read_excel(file_path, sheet_name='Ward_Population')  # assumed sheet name

# ➤ Data Cleaning

# Strip spaces if necessary
stations_df['Station'] = stations_df['Station'].str.strip()
wards_df['Ward'] = wards_df['Ward'].str.strip()

# Drop rows with missing network coordinates for stations
stations_df = stations_df.dropna(subset=['Latitude_network', 'Longitude_network'])

# Add Latitude and Longitude columns to wards_df if they don't exist
# (Incorporated code from cell WUKjnCpVhYiM for robustness)
if 'Latitude' not in wards_df.columns or 'Longitude' not in wards_df.columns:
    # Generate random points around Delhi's center for demo purposes (replace with actual data if available)
    np.random.seed(0) # Use a seed for reproducibility of random points
    wards_df['Latitude'] = 28.6139 + np.random.uniform(-0.1, 0.1, size=len(wards_df))
    wards_df['Longitude'] = 77.2090 + np.random.uniform(-0.1, 0.1, size=len(wards_df))

# Drop rows with missing Latitude, Longitude, or Population for wards
wards_df = wards_df.dropna(subset=['Latitude', 'Longitude', 'Population'])


# ➤ 1. Assign nearest station to each ward

def find_nearest_station(ward_coord):
    min_dist = float('inf')
    nearest_station = None
    for _, row in stations_df.iterrows():
        station_coord = (row['Latitude_network'], row['Longitude_network'])
        dist = geodesic(ward_coord, station_coord).km
        if dist < min_dist:
            min_dist = dist
            nearest_station = row['Station']
    return nearest_station, min_dist

# Apply to each ward
nearest_stations = []
distances = []
for _, row in wards_df.iterrows():
    ward_coord = (row['Latitude'], row['Longitude'])
    station_name, distance = find_nearest_station(ward_coord)
    nearest_stations.append(station_name)
    distances.append(distance)

wards_df['Nearest Station'] = nearest_stations
wards_df['Distance to Station (km)'] = distances

# ➤ 2. Estimate “population served per station”

population_per_station = wards_df.groupby('Nearest Station')['Population'].sum().reset_index()
population_per_station = population_per_station.sort_values(by='Population', ascending=False)

fig = px.bar(population_per_station, x='Nearest Station', y='Population',
             title='Population Served per Station',
             labels={'Population': 'Population'})
fig.update_layout(xaxis_tickangle=-45)
fig.show()

# ➤ 3. Identify underserved areas

# Example: Wards farther than 3 km from any station
underserved_wards = wards_df[wards_df['Distance to Station (km)'] > 3].sort_values(by='Distance to Station (km)', ascending=False)

print("Top underserved areas (ward name, distance):")
print(underserved_wards[['Ward', 'Distance to Station (km)', 'Population']].head())

# Visualize underserved areas on map
fig = px.scatter_mapbox(underserved_wards, lat='Latitude', lon='Longitude', size='Population',
                        hover_name='Ward', hover_data=['Distance to Station (km)'],
                        color='Distance to Station (km)', zoom=10, height=600,
                        title='Underserved Areas (Wards Far from Metro Stations)')

fig.update_layout(mapbox_style='open-street-map')
fig.update_layout(margin={"r":0,"t":50,"l":0,"b":0})
fig.show()

import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

# ➤ Load dataset (assuming stations_df is already loaded)
file_path = '/content/Delhi_Metro_Master (2).xlsx' # Ensure this path is correct
stations_df = pd.read_excel(file_path, sheet_name='Stations') # Reload to be safe

# ➤ Data Cleaning

# Strip spaces from 'Station Layout' and 'Line'
stations_df['Station Layout'] = stations_df['Station Layout'].str.strip()
stations_df['Line'] = stations_df['Line'].str.strip()

# Drop rows with missing Layout information if any
stations_layout_df = stations_df.dropna(subset=['Station Layout'])

# ➤ 1. Elevated vs Underground vs At-grade distribution (overall)

layout_counts = stations_layout_df['Station Layout'].value_counts().reset_index()
layout_counts.columns = ['Layout', 'Count']

plt.figure(figsize=(8,6))
plt.pie(layout_counts['Count'], labels=layout_counts['Layout'], autopct='%1.1f%%', startangle=140)
plt.title('Overall Station Layout Distribution')
plt.tight_layout()
plt.show()

# ➤ 2. Layout preference per line

layout_line = stations_layout_df.groupby(['Line', 'Station Layout']).size().reset_index(name='Count')

fig = px.bar(layout_line, x='Line', y='Count', color='Station Layout',
             title='Layout Preference per Line',
             labels={'Count': 'Number of Stations', 'Station Layout': 'Layout'})
fig.update_layout(barmode='stack', xaxis_tickangle=-45)
fig.show()

# ➤ 3. Spatial Mapping of Station Layouts (Optional, but good for visual context)

# Drop rows missing coordinates if any
stations_map_df = stations_layout_df.dropna(subset=['Latitude_network', 'Longitude_network'])

fig = px.scatter_mapbox(stations_map_df, lat='Latitude_network', lon='Longitude_network', color='Station Layout',
                        hover_name='Station',
                        zoom=10, height=600,
                        title='Spatial Mapping of Station Layouts')

fig.update_layout(mapbox_style='open-street-map')
fig.update_layout(margin={"r":0,"t":50,"l":0,"b":0})
fig.show()

import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd # Ensure pandas is imported

# Assuming wards_df is already loaded and has 'Population' and 'Distance to Station (km)' columns

# Drop rows with missing values in the relevant columns
regression_df = wards_df.dropna(subset=['Population', 'Distance to Station (km)']).copy()

# Define dependent and independent variables
# Dependent variable: Distance to Nearest Station
y = regression_df['Distance to Station (km)']
# Independent variable: Population
X = regression_df['Population']

# Add a constant to the independent variable for the intercept (using statsmodels)
X = sm.add_constant(X)

# Create and fit the regression model
model = sm.OLS(y, X).fit()

# Print the regression summary
print(model.summary())

# ➤ Visualize the relationship
plt.figure(figsize=(10, 6))
sns.scatterplot(data=regression_df, x='Population', y='Distance to Station (km)', alpha=0.6)
plt.title('Ward Population vs. Distance to Nearest Station')
plt.xlabel('Population')
plt.ylabel('Distance to Station (km)')
plt.grid(True)

# Add the regression line to the scatter plot
# Predict the y values using the fitted model
y_pred = model.predict(X)
plt.plot(regression_df['Population'], y_pred, color='red', linewidth=2)

plt.show()

 
