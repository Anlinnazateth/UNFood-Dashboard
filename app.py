import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Streamlit app configuration
st.set_page_config(page_title="Global Food Security Dashboard", layout="wide")
st.title("Global Food Security Dashboard")

# Load the dataset
data = pd.read_csv('food_security.csv')

# Display column names for verification
if st.sidebar.checkbox("contents of the dataset"):
    st.write("content:",data.head())
if st.sidebar.checkbox("Show Dataset Columns"):
    st.write("Dataset Columns:", data.columns.tolist())
if st.sidebar.checkbox("Show data types"):
    st.write("Data types:",data.dtypes)


# Verify required columns
required_columns = ['Year', 'Value', 'Country', 'Area Code (M49)']
optional_columns = ['Region']
missing_columns = [col for col in required_columns if col not in data.columns]

if missing_columns:
    st.error(f"The dataset is missing the following required columns: {missing_columns}")
    st.stop()

# Data preprocessing
# Extract year from range and clean values
data['Year'] = data['Year'].str.split('-').apply(lambda x: int(x[0]) + 1)
data['Value'] = data['Value'].str.replace('<', '').str.replace('>', '').astype(float)

# Remove duplicate entries
data = data.drop_duplicates()

# Sidebar filters
years = sorted(data['Year'].unique())
regions = data['Region'].unique() if 'Region' in data.columns else None
countries = sorted(data['Country'].unique())

selected_year = st.sidebar.selectbox("Select Year", years)
selected_country = st.sidebar.selectbox("Select Country", countries)
selected_countries = st.sidebar.multiselect("Select Countries for Heatmap", countries, default=countries[:5])

# Filter data by the selected year
data_year = data[data['Year'] == selected_year]

# Choropleth map

# Horizontal bar chart
st.subheader("Top Countries with Severe Food Insecurity")
bar_chart_data = data_year.nlargest(10, 'Value')
bar_chart = px.bar(
    bar_chart_data,
    x="Value",
    y="Country",
    orientation="h",
    title="Countries with Highest Food Insecurity",
    color="Value",
    color_continuous_scale="YlOrRd",
    labels={"Value": "Food Insecurity (%)"}
)
st.plotly_chart(bar_chart, use_container_width=True)

# Line chart for a selected country's trend
data_country = data[data['Country'] == selected_country]
st.subheader(f"Food Insecurity Trend for {selected_country}")
line_chart = px.line(
    data_country,
    x="Year",
    y="Value",
    title=f"Food Insecurity Trend: {selected_country}",
    labels={"Value": "Food Insecurity (%)", "Year": "Year"}
)
st.plotly_chart(line_chart, use_container_width=True)

# Histogram
st.subheader(f"Histogram of Food Insecurity Levels ({selected_year})")
histogram = px.histogram(
    data_year,
    x="Value",
    nbins=20,
    title="Distribution of Food Insecurity Levels",
    color_discrete_sequence=["red"],
    labels={"Value": "Food Insecurity (%)"}
)
st.plotly_chart(histogram, use_container_width=True)

# Box plot for regions (if applicable)
if regions is not None:
    st.subheader(f"Food Insecurity Across Regions ({selected_year})")
    box_plot = px.box(
        data_year,
        x="Region",
        y="Value",
        title="Regional Comparison of Food Insecurity",
        color="Region",
        labels={"Value": "Food Insecurity (%)"},
        color_discrete_sequence=px.colors.sequential.YlOrRd
    )
    st.plotly_chart(box_plot, use_container_width=True)

# Heatmap for selected countries across years
st.subheader("Heatmap of Food Insecurity Levels")
data_heatmap = data[data['Country'].isin(selected_countries)]
data_heatmap = data_heatmap.drop_duplicates(subset=['Country', 'Year'])  # Ensure no duplicate entries
data_pivot = data_heatmap.pivot(index="Country", columns="Year", values="Value")
heatmap = go.Figure(data=go.Heatmap(
    z=data_pivot.values,
    x=data_pivot.columns,
    y=data_pivot.index,
    colorscale="YlOrRd",
    colorbar_title="Food Insecurity (%)"
))
heatmap.update_layout(title="Food Insecurity Levels Across Years")
st.plotly_chart(heatmap, use_container_width=True)

