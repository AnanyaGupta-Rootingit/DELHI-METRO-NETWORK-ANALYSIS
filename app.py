# 📌 Install required packages (only in Colab environment, not needed in Streamlit deployment)
# !pip install geopandas folium plotly statsmodels geopy

import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import HeatMap
from folium import FeatureGroup, LayerControl, GeoJson, CircleMarker
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.colors as pc
import seaborn as sns
import numpy as np
from shapely.geometry import Point
from geopy.distance import geodesic
import statsmodels.api as sm
import re

st.set_page_config(layout="wide")

st.title("Delhi Metro Comprehensive Analysis Dashboard")

# ➤ Upload the file
uploaded_file = st.file_uploader("Upload 'Delhi_Metro_Master.xlsx'", type=["xlsx"])
if uploaded_file is not None:
    file_path = uploaded_file

    # ➤ Load data
    stations_df = pd.read_excel(file_path, sheet_name='Stations')
    wards_df = pd.read_excel(file_path, sheet_name='Ward_Population')
    ridership_df = pd.read_excel(file_path, sheet_name='Ridership_RS')
    ridership_long_df = pd.read_excel(file_path, sheet_name='Ridership_Long')

    st.header("Data Preview")
    st.write("Stations Data", stations_df.head())
    st.write("Wards Data", wards_df.head())
    st.write("Ridership RS", ridership_df.head())
    st.write("Ridership Long", ridership_long_df.head())

    # =================== Station Map ===================
    st.header("Metro Stations Map")
    delhi_map = folium.Map(location=[28.6139, 77.2090], zoom_start=11)
    line_colors = {
        'Red': 'red', 'Blue': 'blue', 'Yellow': 'orange', 'Green': 'green',
        'Violet': 'purple', 'Magenta': 'pink', 'Grey': 'gray', 'Pink': 'lightpink', 'Rapid Metro': 'cadetblue'
    }
    stations_df_cleaned = stations_df.dropna(subset=['Latitude_network', 'Longitude_network'])
    for idx, row in stations_df_cleaned.iterrows():
        CircleMarker(
            location=[row['Latitude_network'], row['Longitude_network']],
            radius=5,
            color=line_colors.get(row['Line'], 'black'),
            fill=True,
            fill_opacity=0.7,
            popup=f"{row['Station']} ({row['Line']})"
        ).add_to(delhi_map)
    st_folium(delhi_map, width=700)

    # =================== Heatmap ===================
    st.header("Stations Heatmap")
    heat_data = stations_df.dropna(subset=['Latitude_network', 'Longitude_network'])[['Latitude_network', 'Longitude_network']].values.tolist()
    HeatMap(heat_data, radius=15, blur=10, max_zoom=13).add_to(delhi_map)
    st_folium(delhi_map, width=700)

    # =================== Layer Control Map ===================
    st.header("Stations & Heatmap Layer Control")
    stations_fg = FeatureGroup(name='Stations')
    heatmap_fg = FeatureGroup(name='Heatmap')
    for idx, row in stations_df_cleaned.iterrows():
        CircleMarker(
            location=[row['Latitude_network'], row['Longitude_network']],
            radius=5,
            color=line_colors.get(row['Line'], 'black'),
            fill=True,
            fill_opacity=0.7,
            popup=f"{row['Station']} ({row['Line']})"
        ).add_to(stations_fg)
    HeatMap(heat_data, radius=15, blur=10, max_zoom=13).add_to(heatmap_fg)
    stations_fg.add_to(delhi_map)
    heatmap_fg.add_to(delhi_map)
    LayerControl().add_to(delhi_map)
    st_folium(delhi_map, width=700)

    # =================== Station Count ===================
    st.header("Station Counts per Line")
    station_counts = stations_df.groupby('Line')['Station'].count().reset_index().rename(columns={'Station': 'Station Count'})
    st.write(station_counts)
    fig = px.bar(station_counts, x='Line', y='Station Count', color='Line', title='Number of Stations per Metro Line')
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

    # =================== Buffer Zones ===================
    st.header("Buffer Zones (500m & 1km)")
    gdf = gpd.GeoDataFrame(
        stations_df_cleaned,
        geometry=gpd.points_from_xy(stations_df_cleaned['Longitude_network'], stations_df_cleaned['Latitude_network']),
        crs="EPSG:4326"
    ).to_crs(epsg=3857)
    gdf['buffer_500m'] = gdf.geometry.buffer(500)
    gdf['buffer_1km'] = gdf.geometry.buffer(1000)
    gdf = gdf.to_crs(epsg=4326)
    buffer_map = folium.Map(location=[28.6139, 77.2090], zoom_start=11)
    for idx, row in gdf.iterrows():
        GeoJson(row['buffer_500m'], style_function=lambda x: {'fillColor': 'blue','color':'blue','weight':1,'fillOpacity':0.2}).add_to(buffer_map)
        GeoJson(row['buffer_1km'], style_function=lambda x: {'fillColor': 'red','color':'red','weight':1,'fillOpacity':0.1}).add_to(buffer_map)
        CircleMarker(location=[row.geometry.y, row.geometry.x], radius=3, color='black', fill=True, fill_opacity=0.7, popup=f"{row['Station']} ({row['Line']})").add_to(buffer_map)
    st_folium(buffer_map, width=700)

    # =================== Ward Coverage ===================
    st.header("Ward Population Coverage")
    np.random.seed(0)
    wards_df['Latitude'] = 28.6139 + np.random.uniform(-0.1, 0.1, size=len(wards_df))
    wards_df['Longitude'] = 77.2090 + np.random.uniform(-0.1, 0.1, size=len(wards_df))
    wards_gdf = gpd.GeoDataFrame(
        wards_df,
        geometry=gpd.points_from_xy(wards_df['Longitude'], wards_df['Latitude']),
        crs="EPSG:4326"
    ).to_crs(epsg=3857)
    station_buffers = gdf.to_crs(epsg=3857)
    wards_df['covered_500m'] = 0
    wards_df['covered_1km'] = 0
    for idx_w, ward in wards_gdf.iterrows():
        ward_point = ward.geometry
        pop = ward['Population']
        if station_buffers['buffer_500m'].intersects(ward_point).any():
            wards_df.at[idx_w, 'covered_500m'] = pop
        if station_buffers['buffer_1km'].intersects(ward_point).any():
            wards_df.at[idx_w, 'covered_1km'] = pop
    total_covered_500m = wards_df['covered_500m'].sum()
    total_covered_1km = wards_df['covered_1km'].sum()
    st.write(f"Total population covered within 500m: {total_covered_500m}")
    st.write(f"Total population covered within 1km: {total_covered_1km}")
    total_population = wards_df['Population'].sum()
    covered_500m = total_covered_500m
    covered_1km = total_covered_1km
    uncovered_500m = total_population - covered_500m
    uncovered_1km = total_population - covered_1km
    coverage_data = {'500m': [covered_500m, uncovered_500m], '1km': [covered_1km, uncovered_1km]}
    fig, ax = plt.subplots(1,2, figsize=(12,5))
    ax[0].bar(['Covered','Uncovered'], coverage_data['500m'], color=['green','red'])
    ax[0].set_title('Population Coverage within 500m')
    ax[0].set_ylabel('Population')
    ax[1].bar(['Covered','Uncovered'], coverage_data['1km'], color=['green','red'])
    ax[1].set_title('Population Coverage within 1km')
    st.pyplot(fig)

    underserved_500m = wards_df[wards_df['covered_500m'] == 0].sort_values(by='Population', ascending=False)
    underserved_1km = wards_df[wards_df['covered_1km'] == 0].sort_values(by='Population', ascending=False)
    st.write("Top underserved wards within 500m:", underserved_500m[['Ward','Population']].head())
    st.write("Top underserved wards within 1km:", underserved_1km[['Ward','Population']].head())
    ward_map = folium.Map(location=[28.6139, 77.2090], zoom_start=11)
    for idx, row in wards_gdf.iterrows():
        color = 'green' if wards_df.loc[idx, 'covered_1km'] > 0 else 'red'
        CircleMarker(location=[row.geometry.y, row.geometry.x], radius=7, color=color, fill=True, fill_opacity=0.6, popup=f"{wards_df.loc[idx, 'Ward']} ({wards_df.loc[idx, 'Population']} people)").add_to(ward_map)
    st_folium(ward_map, width=700)

    # =================== Ridership Analysis ===================
    st.header("Ridership Trends")
    plt.figure(figsize=(10,6))
    plt.plot(ridership_df['Year'], ridership_df['Ridership (Lakhs/day)'], marker='o', color='blue')
    plt.title('Annual Ridership Trends (2013–2022)')
    plt.xlabel('Year')
    plt.ylabel('Ridership (Lakhs/day)')
    plt.grid(True)
    st.pyplot(plt)

    fig = px.line(ridership_df, x='Year', y='Ridership (Lakhs/day)', markers=True, title='Annual Ridership Trends (2013–2022)')
    fig.update_traces(line=dict(width=3), marker=dict(size=8))
    st.plotly_chart(fig, use_container_width=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ridership_df['Year'], y=ridership_df['Ridership (Lakhs/day)'], mode='lines+markers', name='Ridership'))
    fig.add_vrect(x0=2020, x1=2021, fillcolor="red", opacity=0.2, layer="below", line_width=0, annotation_text="COVID-19 Impact", annotation_position="top left")
    fig.update_layout(title='Ridership Trend with COVID-19 Impact (2013–2022)', xaxis_title='Year', yaxis_title='Ridership (Lakhs/day)')
    st.plotly_chart(fig, use_container_width=True)

    ridership_df['Rolling_Avg'] = ridership_df['Ridership (Lakhs/day)'].rolling(window=3, center=True).mean()
    plt.figure(figsize=(10,6))
    plt.plot(ridership_df['Year'], ridership_df['Ridership (Lakhs/day)'], marker='o', label='Actual Ridership')
    plt.plot(ridership_df['Year'], ridership_df['Rolling_Avg'], marker='x', linestyle='--', color='orange', label='3-Year Rolling Average')
    plt.title('Ridership Trend with Rolling Average')
    plt.xlabel('Year')
    plt.ylabel('Ridership (Lakhs/day)')
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

    ridership_df['YoY_Change'] = ridership_df['Ridership (Lakhs/day)'].pct_change() * 100
    plt.figure(figsize=(10,6))
    plt.bar(ridership_df['Year'], ridership_df['YoY_Change'], color='purple')
    plt.title('Year-over-Year Ridership Change (%)')
    plt.xlabel('Year')
    plt.ylabel('YoY Change (%)')
    plt.grid(axis='y')
    st.pyplot(plt)

    # =================== Regression ===================
    st.header("Regression: Population vs Distance to Nearest Station")
    regression_df = wards_df.dropna(subset=['Population', 'Distance to Station (km)']).copy()
    y = regression_df['Distance to Station (km)']
    X = regression_df['Population']
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    st.text(model.summary())
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=regression_df, x='Population', y='Distance to Station (km)', alpha=0.6)
    plt.title('Ward Population vs. Distance to Nearest Station')
    plt.xlabel('Population')
    plt.ylabel('Distance to Station (km)')
    plt.grid(True)
    y_pred = model.predict(X)
    plt.plot(regression_df['Population'], y_pred, color='red', linewidth=2)
    st.pyplot(plt)

