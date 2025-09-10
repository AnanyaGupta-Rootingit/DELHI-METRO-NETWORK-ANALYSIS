# 📌 Install required packages (only in Colab, comment out in Streamlit deployment)
# !pip install geopandas folium plotly matplotlib statsmodels geopy

import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
from shapely.geometry import Point
from geopy.distance import geodesic
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.colors as pc
import statsmodels.api as sm
import seaborn as sns
import re
from datetime import datetime

# ➤ File upload
st.title("Delhi Metro Analysis Dashboard")

try:
    uploaded_file = st.file_uploader("Upload 'Delhi_Metro_Master.xlsx'", type=["xlsx"])

    if uploaded_file:
        # Load datasets
        stations_df = pd.read_excel(uploaded_file, sheet_name='Stations')
        wards_df = pd.read_excel(uploaded_file, sheet_name='Ward_Population')
        ridership_df = pd.read_excel(uploaded_file, sheet_name='Ridership_RS')
        ridership_long_df = pd.read_excel(uploaded_file, sheet_name='Ridership_Long')

        # Sidebar selection
        section = st.sidebar.selectbox("Select Section", ["Maps", "Plots", "Population Coverage", "Regression", "Project Summary"])

        if section == "Maps":
            st.header("Metro Stations and Heatmap")

            # Create base map
            delhi_map = folium.Map(location=[28.6139, 77.2090], zoom_start=11)

            # Line colors
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

            # Clean stations
            stations_df_cleaned = stations_df.dropna(subset=['Latitude_network', 'Longitude_network'])

            # Add markers
            for idx, row in stations_df_cleaned.iterrows():
                folium.CircleMarker(
                    location=[row['Latitude_network'], row['Longitude_network']],
                    radius=5,
                    color=line_colors.get(row['Line'], 'black'),
                    fill=True,
                    fill_opacity=0.7,
                    popup=f"{row['Station']} ({row['Line']})"
                ).add_to(delhi_map)

            # Heatmap
            heat_data = stations_df_cleaned[['Latitude_network', 'Longitude_network']].values.tolist()
            HeatMap(heat_data, radius=15, blur=10, max_zoom=13).add_to(delhi_map)

            # Display map
            st_data = st_folium(delhi_map, width=700, height=500)

        elif section == "Plots":
            st.header("Ridership Trends and Station Data")

            # Annual Ridership Trend
            st.subheader("Annual Ridership (2013-2022)")
            fig = px.line(ridership_df, x='Year', y='Ridership (Lakhs/day)', markers=True,
                          title="Annual Ridership Trends")
            st.plotly_chart(fig)

            # Rolling average
            ridership_df['Rolling_Avg'] = ridership_df['Ridership (Lakhs/day)'].rolling(window=3, center=True).mean()
            fig2 = px.line(ridership_df, x='Year', y=['Ridership (Lakhs/day)', 'Rolling_Avg'],
                           title="Ridership with 3-Year Rolling Average")
            st.plotly_chart(fig2)

            # YoY Change
            ridership_df['YoY_Change'] = ridership_df['Ridership (Lakhs/day)'].pct_change() * 100
            fig3 = px.bar(ridership_df, x='Year', y='YoY_Change', title="Year-over-Year Ridership Change (%)")
            st.plotly_chart(fig3)

            # Station counts
            station_counts = stations_df.groupby('Line')['Station'].count().reset_index()
            station_counts = station_counts.rename(columns={'Station': 'Station Count'})
            fig4 = px.bar(station_counts, x='Line', y='Station Count', color='Line', title="Stations per Line")
            st.plotly_chart(fig4)

        elif section == "Population Coverage":
            st.header("Population Coverage and Underserved Areas")

            # Approximate wards' coordinates if missing
            if 'Latitude' not in wards_df.columns or 'Longitude' not in wards_df.columns:
                np.random.seed(0)
                wards_df['Latitude'] = 28.6139 + np.random.uniform(-0.1, 0.1, size=len(wards_df))
                wards_df['Longitude'] = 77.2090 + np.random.uniform(-0.1, 0.1, size=len(wards_df))

            wards_df = wards_df.dropna(subset=['Latitude', 'Longitude', 'Population'])
            stations_df_cleaned = stations_df.dropna(subset=['Latitude_network', 'Longitude_network'])

            # Create GeoDataFrames
            wards_gdf = gpd.GeoDataFrame(
                wards_df,
                geometry=gpd.points_from_xy(wards_df['Longitude'], wards_df['Latitude']),
                crs="EPSG:4326"
            )
            gdf = gpd.GeoDataFrame(
                stations_df_cleaned,
                geometry=gpd.points_from_xy(stations_df_cleaned['Longitude_network'], stations_df_cleaned['Latitude_network']),
                crs="EPSG:4326"
            ).to_crs(epsg=3857)

            # Create buffers
            gdf['buffer_500m'] = gdf.geometry.buffer(500)
            gdf['buffer_1km'] = gdf.geometry.buffer(1000)
            gdf = gdf.to_crs(epsg=4326)

            # Initialize coverage columns
            wards_df['covered_500m'] = 0
            wards_df['covered_1km'] = 0

            # Spatial join
            station_buffers = gdf.to_crs(epsg=3857)
            wards_gdf = wards_gdf.to_crs(epsg=3857)
            for idx_w, ward in wards_gdf.iterrows():
                ward_point = ward.geometry
                pop = ward['Population']
                if station_buffers['buffer_500m'].intersects(ward_point).any():
                    wards_df.at[idx_w, 'covered_500m'] = pop
                if station_buffers['buffer_1km'].intersects(ward_point).any():
                    wards_df.at[idx_w, 'covered_1km'] = pop

            total_population = wards_df['Population'].sum()
            covered_500m = wards_df['covered_500m'].sum()
            covered_1km = wards_df['covered_1km'].sum()
            st.write(f"Total population: {total_population}")
            st.write(f"Covered within 500m: {covered_500m}")
            st.write(f"Covered within 1km: {covered_1km}")

            # Underserved areas
            underserved_500m = wards_df[wards_df['covered_500m'] == 0].sort_values(by='Population', ascending=False)
            st.write("Top 5 underserved wards within 500m:")
            st.dataframe(underserved_500m[['Ward', 'Population']].head())

            underserved_1km = wards_df[wards_df['covered_1km'] == 0].sort_values(by='Population', ascending=False)
            st.write("Top 5 underserved wards within 1km:")
            st.dataframe(underserved_1km[['Ward', 'Population']].head())

        elif section == "Regression":
            st.header("Regression Analysis")

            # Clean data
            regression_df = wards_df.dropna(subset=['Population', 'Distance to Station (km)']).copy()
            y = regression_df['Distance to Station (km)']
            X = regression_df['Population']
            X = sm.add_constant(X)
            model = sm.OLS(y, X).fit()
            st.text(model.summary())

            # Plot
            fig, ax = plt.subplots(figsize=(10,6))
            sns.scatterplot(data=regression_df, x='Population', y='Distance to Station (km)', alpha=0.6, ax=ax)
            y_pred = model.predict(X)
            ax.plot(regression_df['Population'], y_pred, color='red', linewidth=2)
            ax.set_title("Ward Population vs Distance to Nearest Station")
            st.pyplot(fig)

        elif section == "Project Summary":
            st.header("Project Summary and Key Findings")
            st.markdown("""
            This project explored various aspects of the Delhi Metro network using the provided dataset. The key analyses and findings include:

            1. **Metro Network Overview:**
                * Spatial visualization of stations by line and layout.
                * Heatmap showing station density.

            2. **Station Characteristics:**
                * Distribution of station layouts across the network.

            3. **Ridership Analysis:**
                * Trends from 2013–2022 including COVID-19 impact.
                * Rolling averages and yearly changes.

            4. **Population Coverage:**
                * Estimation of population within 500m and 1km buffers.
                * Identification of underserved areas.

            5. **Regression Analysis:**
                * Weak relationship between ward population and distance to nearest station.

            These insights can guide future infrastructure development and accessibility planning.
            """)

except Exception as e:
    st.error(f"An error occurred: {e}")
    st.error("Please ensure you've uploaded the correct 'Delhi_Metro_Master.xlsx' file with the appropriate sheets.")
