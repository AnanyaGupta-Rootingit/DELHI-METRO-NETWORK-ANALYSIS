import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.colors as pc
import re
from geopy.distance import geodesic
import numpy as np
import statsmodels.api as sm
import seaborn as sns
import io

# Set page configuration
st.set_page_config(layout="wide", page_title="Delhi Metro Analysis Dashboard")

st.title("Delhi Metro Network Analysis")

# --- File Upload ---
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload Delhi_Metro_Master.xlsx", type="xlsx")

if uploaded_file is not None:
    try:
        # Load data from the uploaded Excel file
        stations_df = pd.read_excel(uploaded_file, sheet_name='Stations')
        wards_df = pd.read_excel(uploaded_file, sheet_name='Ward_Population')
        ridership_df = pd.read_excel(uploaded_file, sheet_name='Ridership_RS')
        ridership_long_df = pd.read_excel(uploaded_file, sheet_name='Ridership_Long')

        st.sidebar.success("Data loaded successfully!")

        # --- Data Cleaning (Replicate cleaning from notebook) ---

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

        # Strip spaces from 'Station Layout' and 'Station' and 'Ward'
        stations_df['Station Layout'] = stations_df['Station Layout'].str.strip()
        stations_df['Station'] = stations_df['Station'].str.strip()
        wards_df['Ward'] = wards_df['Ward'].str.strip()

        # Drop rows with missing network coordinates for stations for mapping/geospatial analysis
        stations_gdf_cleaned = stations_df.dropna(subset=['Latitude_network', 'Longitude_network']).copy()

        # Add Latitude and Longitude columns to wards_df if they don't exist (using random for demo)
        if 'Latitude' not in wards_df.columns or 'Longitude' not in wards_df.columns:
            # Generate random points around Delhi's center for demo purposes (replace with actual data if available)
            np.random.seed(0) # Use a seed for reproducibility of random points
            wards_df['Latitude'] = 28.6139 + np.random.uniform(-0.1, 0.1, size=len(wards_df))
            wards_df['Longitude'] = 77.2090 + np.random.uniform(-0.1, 0.1, size=len(wards_df))

        # Drop rows with missing Latitude, Longitude, or Population for wards
        wards_df_cleaned = wards_df.dropna(subset=['Latitude', 'Longitude', 'Population']).copy()

        # --- Calculate Nearest Station and Distance (Moved here to be available for all sections) ---
        if not wards_df_cleaned.empty and not stations_gdf_cleaned.empty:
            def find_nearest_station(ward_coord, stations_df):
                min_dist = float('inf')
                nearest_station = None
                for _, row in stations_df.iterrows():
                    station_coord = (row['Latitude_network'], row['Longitude_network'])
                    dist = geodesic(ward_coord, station_coord).km
                    if dist < min_dist:
                        min_dist = dist
                        nearest_station = row['Station']
                return nearest_station, min_dist

            nearest_stations = []
            distances = []
            for idx, row in wards_df_cleaned.iterrows():
                ward_coord = (row['Latitude'], row['Longitude'])
                station_name, distance = find_nearest_station(ward_coord, stations_gdf_cleaned)
                nearest_stations.append(station_name)
                distances.append(distance)

            wards_df_cleaned['Nearest Station'] = nearest_stations
            wards_df_cleaned['Distance to Station (km)'] = distances
        else:
            st.warning("Ward population or station data is not available to calculate nearest station distances.")


        # --- Navigation Sidebar ---
        st.sidebar.header("Analysis Sections")
        section = st.sidebar.radio("Go to", [
            "Network Overview",
            "Station Characteristics",
            "Ridership Analysis",
            "Population Coverage",
            "Regression Analysis",
            "Project Summary"
        ])

        # --- Display Sections ---

        if section == "Network Overview":
            st.header("Metro Network Overview")

            # Base Map
            st.subheader("Delhi Metro Map")
            m = folium.Map(location=[28.6139, 77.2090], zoom_start=11)

            # Add stations as markers
            line_colors = {
                'Red line': 'red', 'Blue line': 'blue', 'Yellow line': 'orange', 'Green line': 'green',
                'Violet line': 'purple', 'Magenta line': 'pink', 'Grey line': 'gray', 'Pink line': 'lightpink',
                'Rapid Metro': 'cadetblue', 'Aqua line': 'cyan', 'Orange line': 'darkorange',
                'Blue Line': 'blue', 'Green Line': 'green', 'Pink Line': 'lightpink', 'Red Line': 'red',
                'Violet Line': 'purple', 'Yellow Line': 'orange', 'Magenta Line': 'pink', 'Grey Line': 'gray',
                'Aqua Line': 'cyan', 'Orange Line': 'darkorange', 'Blue Line branch': 'lightblue', 'Green Line branch': 'lightgreen',
                'Grey Line branch': 'lightgray' # Added variations and branches
            }


            for idx, row in stations_gdf_cleaned.iterrows():
                folium.CircleMarker(
                    location=[row['Latitude_network'], row['Longitude_network']],
                    radius=5,
                    color=line_colors.get(row['Line'], 'black'),
                    fill=True,
                    fill_opacity=0.7,
                    popup=f"{row['Station']} ({row['Line']})"
                ).add_to(m)

            # Add HeatMap layer
            heat_data = stations_gdf_cleaned[['Latitude_network', 'Longitude_network']].values.tolist()
            HeatMap(heat_data, radius=15, blur=10, max_zoom=13).add_to(m)

            st_folium(m, width=700, height=500)

            st.subheader("Interactive Station and Buffer Map")

            # Interactive controls
            unique_lines = stations_df['Line'].unique().tolist()
            selected_lines = st.multiselect("Select Lines:", unique_lines, default=unique_lines)

            unique_layouts = stations_df['Station Layout'].dropna().unique().tolist()
            selected_layouts = st.multiselect("Select Layouts:", unique_layouts, default=unique_layouts)

            buffer_distance = st.radio("Select Buffer Distance (meters):", [500, 1000])

            # Filter stations based on selection
            filtered_stations = stations_df[
                (stations_df['Line'].isin(selected_lines)) &
                (stations_df['Station Layout'].isin(selected_layouts))
            ].dropna(subset=['Latitude_network', 'Longitude_network']).copy()

            if not filtered_stations.empty:
                # Convert to GeoDataFrame and create buffers
                gdf_filtered = gpd.GeoDataFrame(
                    filtered_stations,
                    geometry=gpd.points_from_xy(filtered_stations['Longitude_network'], filtered_stations['Latitude_network']),
                    crs="EPSG:4326"
                ).to_crs(epsg=3857)

                gdf_filtered['buffer'] = gdf_filtered.geometry.buffer(buffer_distance)
                gdf_filtered = gdf_filtered.to_crs(epsg=4326)

                # Create map
                buffer_map = folium.Map(location=[28.6139, 77.2090], zoom_start=11)

                # Add buffers
                for idx, row in gdf_filtered.iterrows():
                    folium.GeoJson(row['buffer'],
                                   style_function=lambda x: {'fillColor': 'blue', 'color': 'blue', 'weight': 1, 'fillOpacity': 0.2}
                                  ).add_to(buffer_map)

                # Add station markers
                for idx, row in gdf_filtered.iterrows():
                    folium.CircleMarker(
                        location=[row.geometry.y, row.geometry.x],
                        radius=3,
                        color='black',
                        fill=True,
                        fill_opacity=0.7,
                        popup=f"{row['Station']} ({row['Line']})"
                    ).add_to(buffer_map)

                st_folium(buffer_map, width=700, height=500)
            else:
                st.warning("No stations match the selected criteria.")


        elif section == "Station Characteristics":
            st.header("Station Characteristics")

            st.subheader("Overall Station Layout Distribution")
            layout_counts = stations_df['Station Layout'].value_counts().reset_index()
            layout_counts.columns = ['Layout', 'Count']
            fig_pie, ax_pie = plt.subplots(figsize=(8,6))
            ax_pie.pie(layout_counts['Count'], labels=layout_counts['Layout'], autopct='%1.1f%%', startangle=140)
            ax_pie.set_title('Overall Station Layout Distribution')
            st.pyplot(fig_pie)

            st.subheader("Layout Preference per Line")
            layout_line = stations_df.groupby(['Line', 'Station Layout']).size().reset_index(name='Count')
            fig_bar = px.bar(layout_line, x='Line', y='Count', color='Station Layout',
                             title='Layout Preference per Line',
                             labels={'Count': 'Number of Stations', 'Station Layout': 'Layout'})
            fig_bar.update_layout(barmode='stack', xaxis_tickangle=-45)
            st.plotly_chart(fig_bar)

            st.subheader("Spatial Mapping of Station Layouts")
            stations_map_df = stations_df.dropna(subset=['Latitude_network', 'Longitude_network']).copy()
            fig_map = px.scatter_mapbox(stations_map_df, lat='Latitude_network', lon='Longitude_network', color='Station Layout',
                                        hover_name='Station',
                                        zoom=10, height=600,
                                        title='Spatial Mapping of Station Layouts')
            fig_map.update_layout(mapbox_style='open-street-map')
            fig_map.update_layout(margin={"r":0,"t":50,"l":0,"b":0})
            st.plotly_chart(fig_map)


        elif section == "Ridership Analysis":
            st.header("Ridership Analysis")

            st.subheader("Annual Ridership Trends (2018–2022)")
            fig_ridership_line = px.line(ridership_df, x='Year', y='Ridership (Lakhs/day)', markers=True,
                                          title='Annual Ridership Trends (2018–2022)',
                                          labels={'Ridership (Lakhs/day)': 'Ridership', 'Year': 'Year'})
            fig_ridership_line.update_traces(line=dict(width=3), marker=dict(size=8))
            st.plotly_chart(fig_ridership_line)

            st.subheader("Ridership Trend with COVID-19 Impact")
            fig_covid = go.Figure()
            fig_covid.add_trace(go.Scatter(x=ridership_df['Year'], y=ridership_df['Ridership (Lakhs/day)'],
                                             mode='lines+markers', name='Ridership'))
            fig_covid.add_vrect(x0=2020, x1=2021, fillcolor="red", opacity=0.2,
                                  layer="below", line_width=0,
                                  annotation_text="COVID-19 Impact", annotation_position="top left")
            fig_covid.update_layout(title='Ridership Trend with COVID-19 Impact (2018–2022)',
                                      xaxis_title='Year',
                                      yaxis_title='Ridership (Lakhs/day)',
                                      xaxis=dict(tickmode='linear'))
            st.plotly_chart(fig_covid)

            st.subheader("Ridership Trend with Rolling Average (3-Year)")
            ridership_df['Rolling_Avg'] = ridership_df['Ridership (Lakhs/day)'].rolling(window=3, center=True).mean()
            fig_rolling, ax_rolling = plt.subplots(figsize=(10,6))
            ax_rolling.plot(ridership_df['Year'], ridership_df['Ridership (Lakhs/day)'], marker='o', label='Actual Ridership')
            ax_rolling.plot(ridership_df['Year'], ridership_df['Rolling_Avg'], marker='x', linestyle='--', color='orange', label='3-Year Rolling Average')
            ax_rolling.set_title('Ridership Trend with Rolling Average', fontsize=14)
            ax_rolling.set_xlabel('Year', fontsize=12)
            ax_rolling.set_ylabel('Ridership (Lakhs/day)', fontsize=12)
            ax_rolling.legend()
            ax_rolling.grid(True)
            st.pyplot(fig_rolling)

            st.subheader("Year-over-Year Ridership Change (%)")
            ridership_df['YoY_Change'] = ridership_df['Ridership (Lakhs/day)'].pct_change() * 100
            fig_yoy, ax_yoy = plt.subplots(figsize=(10,6))
            ax_yoy.bar(ridership_df['Year'], ridership_df['YoY_Change'], color='purple')
            ax_yoy.set_title('Year-over-Year Ridership Change (%)', fontsize=14)
            ax_yoy.set_xlabel('Year', fontsize=12)
            ax_yoy.set_ylabel('YoY Change (%)', fontsize=12)
            ax_yoy.grid(axis='y')
            st.pyplot(fig_yoy)


        elif section == "Population Coverage":
            st.header("Population Coverage and Underserved Areas")

            # Ensure wards_df_cleaned is used for this section
            if not wards_df_cleaned.empty and not stations_gdf_cleaned.empty and 'Distance to Station (km)' in wards_df_cleaned.columns:

                st.subheader("Population Served per Station (Estimated)")
                population_per_station = wards_df_cleaned.groupby('Nearest Station')['Population'].sum().reset_index()
                population_per_station = population_per_station.sort_values(by='Population', ascending=False)
                fig_pop_station = px.bar(population_per_station.head(20), x='Nearest Station', y='Population', # Display top 20
                                         title='Top 20 Stations by Estimated Population Served',
                                         labels={'Population': 'Population'})
                fig_pop_station.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig_pop_station)


                st.subheader("Population Coverage within Buffer Zones")

                # Convert stations to GeoDataFrame for buffer analysis
                stations_geo_3857 = gpd.GeoDataFrame(
                    stations_gdf_cleaned,
                    geometry=gpd.points_from_xy(stations_gdf_cleaned['Longitude_network'], stations_gdf_cleaned['Latitude_network']),
                    crs="EPSG:4326"
                ).to_crs(epsg=3857)

                # Create buffer zones
                stations_geo_3857['buffer_500m'] = stations_geo_3857.geometry.buffer(500)
                stations_geo_3857['buffer_1km'] = stations_geo_3857.geometry.buffer(1000)

                # Convert wards to EPSG:3857 for spatial operations
                wards_geo_3857 = gpd.GeoDataFrame(
                    wards_df_cleaned,
                    geometry=gpd.points_from_xy(wards_df_cleaned['Longitude'], wards_df_cleaned['Latitude']),
                    crs="EPSG:4326"
                ).to_crs(epsg=3857)

                # Initialize population sums
                wards_df_cleaned['covered_500m'] = 0
                wards_df_cleaned['covered_1km'] = 0

                # Perform spatial intersection
                # Use spatial join for efficiency
                wards_with_500m_coverage = gpd.sjoin(wards_geo_3857, stations_geo_3857[['buffer_500m', 'geometry']], how="inner", predicate="intersects")
                wards_with_1km_coverage = gpd.sjoin(wards_geo_3857, stations_geo_3857[['buffer_1km', 'geometry']], how="inner", predicate="intersects")


                # Update coverage status in wards_df_cleaned
                wards_df_cleaned['covered_500m'] = np.where(wards_df_cleaned.index.isin(wards_with_500m_coverage.index), wards_df_cleaned['Population'], 0)
                wards_df_cleaned['covered_1km'] = np.where(wards_df_cleaned.index.isin(wards_with_1km_coverage.index), wards_df_cleaned['Population'], 0)


                # Sum total population covered
                total_population = wards_df_cleaned['Population'].sum()
                covered_500m = wards_df_cleaned['covered_500m'].sum()
                covered_1km = wards_df_cleaned['covered_1km'].sum()
                uncovered_500m = total_population - covered_500m
                uncovered_1km = total_population - covered_1km

                coverage_data = {
                    '500m': [covered_500m, uncovered_500m],
                    '1km': [covered_1km, uncovered_1km]
                }

                col1, col2 = st.columns(2)

                with col1:
                    st.write("Population Coverage within 500m:")
                    fig_cov_500, ax_cov_500 = plt.subplots(figsize=(6,5))
                    ax_cov_500.bar(['Covered', 'Uncovered'], coverage_data['500m'], color=['green', 'red'])
                    ax_cov_500.set_title('Population Coverage within 500m')
                    ax_cov_500.set_ylabel('Population')
                    st.pyplot(fig_cov_500)

                with col2:
                    st.write("Population Coverage within 1km:")
                    fig_cov_1km, ax_cov_1km = plt.subplots(figsize=(6,5))
                    ax_cov_1km.bar(['Covered', 'Uncovered'], coverage_data['1km'], color=['green', 'red'])
                    ax_cov_1km.set_title('Population Coverage within 1km')
                    st.pyplot(fig_cov_1km)

                st.subheader("Underserved Areas")
                underserved_500m = wards_df_cleaned[wards_df_cleaned['covered_500m'] == 0].sort_values(by='Population', ascending=False)
                underserved_1km = wards_df_cleaned[wards_df_cleaned['covered_1km'] == 0].sort_values(by='Population', ascending=False)

                st.write("Top 5 underserved wards within 500m:")
                st.dataframe(underserved_500m[['Ward', 'Population', 'Distance to Station (km)']].head())

                st.write("Top 5 underserved wards within 1km:")
                st.dataframe(underserved_1km[['Ward', 'Population', 'Distance to Station (km)']].head())

                st.subheader("Map of Underserved Wards (within 1km buffer)")

                # Visualize underserved areas on map
                fig_underserved_map = px.scatter_mapbox(wards_df_cleaned, lat='Latitude', lon='Longitude', size='Population',
                                                        hover_name='Ward', hover_data=['Distance to Station (km)', 'covered_1km'],
                                                        color=wards_df_cleaned['covered_1km'] > 0,
                                                        color_discrete_map={True: 'green', False: 'red'},
                                                        labels={'color': 'Coverage (1km)'}, # Corrected labels for legend
                                                        zoom=10, height=600,
                                                        title='Wards by Metro Coverage within 1km')

                fig_underserved_map.update_layout(mapbox_style='open-street-map')
                fig_underserved_map.update_layout(margin={"r":0,"t":50,"l":0,"b":0})
                st.plotly_chart(fig_underserved_map)

            else:
                st.warning("Ward population or station data is not available to perform this analysis.")


        elif section == "Regression Analysis":
            st.header("Regression Analysis: Ward Population vs. Distance to Nearest Station")

            # Ensure wards_df_cleaned is populated with 'Distance to Station (km)'
            if 'Distance to Station (km)' in wards_df_cleaned.columns and not wards_df_cleaned.empty:
                 # Drop rows with missing values in the relevant columns
                regression_df = wards_df_cleaned.dropna(subset=['Population', 'Distance to Station (km)']).copy()

                if not regression_df.empty:
                    # Define dependent and independent variables
                    y = regression_df['Distance to Station (km)']
                    X = regression_df['Population']

                    # Add a constant
                    X = sm.add_constant(X)

                    # Create and fit the regression model
                    model = sm.OLS(y, X).fit()

                    st.subheader("Regression Model Summary")
                    # Use st.text or st.write for the summary as st.dataframe doesn't format it well
                    st.text(model.summary())

                    st.subheader("Ward Population vs. Distance to Nearest Station with Regression Line")
                    fig_reg, ax_reg = plt.subplots(figsize=(10, 6))
                    sns.scatterplot(data=regression_df, x='Population', y='Distance to Station (km)', alpha=0.6, ax=ax_reg)
                    ax_reg.set_title('Ward Population vs. Distance to Nearest Station')
                    ax_reg.set_xlabel('Population')
                    ax_reg.set_ylabel('Distance to Station (km)')
                    ax_reg.grid(True)

                    # Add the regression line
                    y_pred = model.predict(X)
                    ax_reg.plot(regression_df['Population'], y_pred, color='red', linewidth=2)

                    st.pyplot(fig_reg)
                else:
                    st.warning("Not enough data with both population and distance information for regression analysis.")
            else:
                st.warning("Nearest station distance data is not available for regression analysis. Please ensure the 'Population Coverage' section is visited after uploading the data to calculate distances.")


        elif section == "Project Summary":
            st.header("Project Summary and Key Findings")
            st.markdown("""
            This project explored various aspects of the Delhi Metro network using the provided dataset. The key analyses and findings include:

            1.  **Metro Network Overview:**
                *   We visualized the spatial distribution of metro stations across Delhi, colored by their respective lines.
                *   A heatmap was created to show the density of metro stations.
                *   An interactive map was developed to filter stations by line and layout, and visualize buffer zones (500m and 1km) around selected stations.
            2.  **Station Characteristics:**
                *   Analysis of station layouts revealed the distribution of Elevated, Underground, and At-Grade stations both overall and per metro line.
            3.  **Ridership Analysis:**
                *   We examined the annual ridership trends from 2018-19 to 2021-22, observing the significant impact of the COVID-19 pandemic on ridership figures.
                *   Rolling averages and year-over-year percentage changes in ridership were calculated and visualized to better understand the trends and recovery.
            4.  **Population Coverage and Underserved Areas:**
                *   By approximating ward locations and calculating the distance to the nearest metro station, we estimated the population covered within 500m and 1km buffer zones.
                *   We identified and listed the top underserved wards based on their distance from the nearest station.
                *   A map was generated to visualize wards colored by their coverage status within the 1km buffer.
            5.  **Regression Analysis:**
                *   A linear regression analysis was performed to explore the relationship between ward population and the distance to the nearest metro station.
                *   The analysis indicated a very weak linear relationship, suggesting that based on this model and the approximate ward locations, population size alone is not a strong predictor of a ward's proximity to a metro station.

            Overall, these analyses provide insights into the physical structure, operational performance (ridership trends), and spatial accessibility of the Delhi Metro network in relation to the population distribution of Delhi's wards. Further analysis could benefit from more precise geographical data for wards and more detailed ridership information linked to specific lines or stations.
            """)

    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.error("Please ensure you have uploaded the correct 'Delhi_Metro_Master.xlsx' file and that the sheet names ('Stations', 'Ward_Population', 'Ridership_RS', 'Ridership_Long') are correct.")

else:
    st.info("Please upload the 'Delhi_Metro_Master.xlsx' file to get started.")
