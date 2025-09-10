import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import HeatMap
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.colors as pc
import re
import numpy as np
from shapely.geometry import Point
from geopy.distance import geodesic
import statsmodels.api as sm
import seaborn as sns
from streamlit_folium import st_folium

st.set_page_config(layout="wide", page_title="Delhi Metro Analysis")

try:
    st.header("Upload Dataset")
    uploaded_file = st.file_uploader("Upload 'Delhi_Metro_Master.xlsx'", type="xlsx")
    
    if uploaded_file is not None:
        file_path = uploaded_file
        
        # 📌 Load the datasets
        stations_df = pd.read_excel(file_path, sheet_name='Stations')
        wards_df = pd.read_excel(file_path, sheet_name='Ward_Population')
        ridership_df = pd.read_excel(file_path, sheet_name='Ridership_RS')
        ridership_long_df = pd.read_excel(file_path, sheet_name='Ridership_Long')
        
        # ➤ Display first few rows
        st.subheader("Stations and Wards Preview")
        st.write(stations_df.head())
        st.write(wards_df.head())
        
        # ➤ Map with Station Markers
        st.subheader("Metro Stations Map")
        delhi_map = folium.Map(location=[28.6139, 77.2090], zoom_start=11)
        line_colors = {
            'Red': 'red', 'Blue': 'blue', 'Yellow': 'orange', 'Green': 'green',
            'Violet': 'purple', 'Magenta': 'pink', 'Grey': 'gray',
            'Pink': 'lightpink', 'Rapid Metro': 'cadetblue'
        }
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
        st_folium(delhi_map, width=700)
        
        # ➤ Heatmap
        st.subheader("Metro Stations Heatmap")
        heat_data = stations_df.dropna(subset=['Latitude_network', 'Longitude_network'])[['Latitude_network', 'Longitude_network']].values.tolist()
        HeatMap(heat_data, radius=15, blur=10, max_zoom=13).add_to(delhi_map)
        st_folium(delhi_map, width=700)
        
        # ➤ Layer Control Map
        st.subheader("Interactive Map with Layers")
        stations_fg = folium.FeatureGroup(name='Stations')
        heatmap_fg = folium.FeatureGroup(name='Heatmap')
        for idx, row in stations_df_cleaned.iterrows():
            folium.CircleMarker(
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
        folium.LayerControl().add_to(delhi_map)
        st_folium(delhi_map, width=700)
        
        # ➤ Station Count Plot
        st.subheader("Number of Stations per Line")
        station_counts = stations_df.groupby('Line')['Station'].count().reset_index().rename(columns={'Station': 'Station Count'})
        st.write(station_counts)
        plt.figure(figsize=(10,6))
        plt.bar(station_counts['Line'], station_counts['Station Count'], color='skyblue')
        plt.xticks(rotation=45, ha='right')
        plt.title('Number of Stations per Metro Line')
        plt.xlabel('Line')
        plt.ylabel('Station Count')
        plt.tight_layout()
        st.pyplot(plt)
        fig = px.bar(station_counts, x='Line', y='Station Count', color='Line', title='Number of Stations per Metro Line', labels={'Line': 'Metro Line', 'Station Count': 'Stations'})
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig)
        
        # ➤ Buffer Zone Map
        st.subheader("Buffer Zones Around Stations")
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
            folium.GeoJson(row['buffer_500m'], style_function=lambda x: {'fillColor': 'blue', 'color': 'blue', 'weight': 1, 'fillOpacity': 0.2}).add_to(buffer_map)
            folium.GeoJson(row['buffer_1km'], style_function=lambda x: {'fillColor': 'red', 'color': 'red', 'weight': 1, 'fillOpacity': 0.1}).add_to(buffer_map)
            folium.CircleMarker(location=[row.geometry.y, row.geometry.x], radius=3, color='black', fill=True, fill_opacity=0.7, popup=f"{row['Station']} ({row['Line']})").add_to(buffer_map)
        st_folium(buffer_map, width=700)
        
        # ➤ Population Coverage
        st.subheader("Population Coverage by Buffer Zones")
        np.random.seed(0)
        wards_df['Latitude'] = 28.6139 + np.random.uniform(-0.1, 0.1, size=len(wards_df))
        wards_df['Longitude'] = 77.2090 + np.random.uniform(-0.1, 0.1, size=len(wards_df))
        wards_gdf = gpd.GeoDataFrame(wards_df, geometry=gpd.points_from_xy(wards_df['Longitude'], wards_df['Latitude']), crs="EPSG:4326").to_crs(epsg=3857)
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
        
        # ➤ Coverage Plot
        total_population = wards_df['Population'].sum()
        covered_500m = wards_df['covered_500m'].sum()
        covered_1km = wards_df['covered_1km'].sum()
        uncovered_500m = total_population - covered_500m
        uncovered_1km = total_population - covered_1km
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        ax[0].bar(['Covered', 'Uncovered'], [covered_500m, uncovered_500m], color=['green', 'red'])
        ax[0].set_title('Population Coverage within 500m')
        ax[0].set_ylabel('Population')
        ax[1].bar(['Covered', 'Uncovered'], [covered_1km, uncovered_1km], color=['green', 'red'])
        ax[1].set_title('Population Coverage within 1km')
        plt.tight_layout()
        st.pyplot(fig)
        
        underserved_500m = wards_df[wards_df['covered_500m'] == 0].sort_values(by='Population', ascending=False)
        underserved_1km = wards_df[wards_df['covered_1km'] == 0].sort_values(by='Population', ascending=False)
        st.subheader("Top underserved wards within 500m")
        st.write(underserved_500m[['Ward', 'Population']].head())
        st.subheader("Top underserved wards within 1km")
        st.write(underserved_1km[['Ward', 'Population']].head())
        
        # ➤ Underserved Map
        ward_map = folium.Map(location=[28.6139, 77.2090], zoom_start=11)
        for idx, row in wards_gdf.iterrows():
            color = 'green' if wards_df.loc[idx, 'covered_1km'] > 0 else 'red'
            folium.CircleMarker(location=[row.geometry.y, row.geometry.x], radius=7, color=color, fill=True, fill_opacity=0.6, popup=f"{wards_df.loc[idx, 'Ward']} ({wards_df.loc[idx, 'Population']} people)").add_to(ward_map)
        st_folium(ward_map, width=700)
        
        # ➤ Interactive Buffer Map
        st.subheader("Interactive Map with Filters")
        unique_lines = stations_df['Line'].unique().tolist()
        line_selector = st.multiselect("Select Lines", unique_lines, default=unique_lines)
        unique_layouts = stations_df['Station Layout'].dropna().unique().tolist()
        layout_selector = st.multiselect("Select Layouts", unique_layouts, default=unique_layouts)
        buffer_distance = st.radio("Select Buffer Distance", [500, 1000], index=0)
        
        # Filter stations
        filtered_stations = stations_df[(stations_df['Line'].isin(line_selector)) & (stations_df['Station Layout'].isin(layout_selector))].dropna(subset=['Latitude_network', 'Longitude_network'])
        gdf_filtered = gpd.GeoDataFrame(filtered_stations, geometry=gpd.points_from_xy(filtered_stations['Longitude_network'], filtered_stations['Latitude_network']), crs="EPSG:4326").to_crs(epsg=3857)
        gdf_filtered['buffer'] = gdf_filtered.geometry.buffer(buffer_distance)
        gdf_filtered = gdf_filtered.to_crs(epsg=4326)
        map_filtered = folium.Map(location=[28.6139, 77.2090], zoom_start=11)
        for idx, row in gdf_filtered.iterrows():
            folium.GeoJson(row['buffer'], style_function=lambda x: {'fillColor': 'blue', 'color': 'blue', 'weight': 1, 'fillOpacity': 0.2}).add_to(map_filtered)
            folium.CircleMarker(location=[row.geometry.y, row.geometry.x], radius=5, color='black', fill=True, fill_opacity=0.7, popup=f"{row['Station']} ({row['Line']})").add_to(map_filtered)
        st_folium(map_filtered, width=700)
        
        # ➤ Ridership Trends
        st.subheader("Annual Ridership Trends")
        plt.figure(figsize=(10,6))
        plt.plot(ridership_df['Year'], ridership_df['Ridership (Lakhs/day)'], marker='o', color='blue')
        plt.title('Annual Ridership Trends (2013–2022)', fontsize=14)
        plt.xlabel('Year', fontsize=12)
        plt.ylabel('Ridership (Lakhs/day)', fontsize=12)
        plt.grid(True)
        plt.xticks(ridership_df['Year'])
        plt.tight_layout()
        st.pyplot(plt)
        fig = px.line(ridership_df, x='Year', y='Ridership (Lakhs/day)', markers=True, title='Annual Ridership Trends (2013–2022)', labels={'Ridership (Lakhs/day)': 'Ridership', 'Year': 'Year'})
        fig.update_traces(line=dict(width=3), marker=dict(size=8))
        st.plotly_chart(fig)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ridership_df['Year'], y=ridership_df['Ridership (Lakhs/day)'], mode='lines+markers', name='Ridership'))
        fig.add_vrect(x0=2020, x1=2021, fillcolor="red", opacity=0.2, layer="below", line_width=0, annotation_text="COVID-19 Impact", annotation_position="top left")
        fig.update_layout(title='Ridership Trend with COVID-19 Impact (2013–2022)', xaxis_title='Year', yaxis_title='Ridership (Lakhs/day)', xaxis=dict(tickmode='linear'))
        st.plotly_chart(fig)
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
        st.pyplot(plt)
        ridership_df['YoY_Change'] = ridership_df['Ridership (Lakhs/day)'].pct_change() * 100
        plt.figure(figsize
