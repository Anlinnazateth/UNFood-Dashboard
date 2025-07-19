"""Global Food Security Dashboard.

Interactive Streamlit dashboard for visualizing UN FAO food insecurity data,
with built-in ML modeling capabilities.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Global Food Security Dashboard",
    page_icon="🌍",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Data loading and preprocessing
# ---------------------------------------------------------------------------
REQUIRED_COLUMNS = ["Year", "Value", "Country", "Area Code (M49)"]


@st.cache_data
def load_data(path: str = "food_security.csv") -> pd.DataFrame:
    """Load and preprocess the food security dataset."""
    try:
        data = pd.read_csv(path, encoding="utf-8-sig")
    except FileNotFoundError:
        st.error(f"Dataset not found at `{path}`. Place the CSV in the project root.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        st.stop()

    missing = [c for c in REQUIRED_COLUMNS if c not in data.columns]
    if missing:
        st.error(f"Dataset is missing required columns: {missing}")
        st.stop()

    # Extract midpoint year from ranges like "2000-2002"
    data["Year"] = (
        data["Year"]
        .astype(str)
        .str.split("-")
        .apply(lambda x: int(x[0]) + 1 if len(x) == 2 else int(x[0]))
    )

    # Clean numeric values (remove < > characters)
    data["Value"] = (
        data["Value"]
        .astype(str)
        .str.replace("<", "", regex=False)
        .str.replace(">", "", regex=False)
    )
    data["Value"] = pd.to_numeric(data["Value"], errors="coerce")
    data = data.dropna(subset=["Value"])
    data = data.drop_duplicates()
    return data


data = load_data()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("🌍 Navigation")
page = st.sidebar.radio("Choose a section:", ["Overview", "Visualizations", "Modeling"])

# Common filters
years = sorted(data["Year"].unique())
countries = sorted(data["Country"].unique())

selected_year = st.sidebar.selectbox("Select Year", years, index=len(years) - 1)
selected_country = st.sidebar.selectbox("Select Country", countries)

# ---------------------------------------------------------------------------
# Overview page
# ---------------------------------------------------------------------------
if page == "Overview":
    st.title("Global Food Security Dashboard")
    st.markdown("Explore global food insecurity trends using UN FAO data.")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Countries", data["Country"].nunique())
    col2.metric("Year Range", f"{data['Year'].min()}–{data['Year'].max()}")
    col3.metric("Data Points", f"{len(data):,}")
    col4.metric("Indicators", data["Item"].nunique() if "Item" in data.columns else "N/A")

    st.subheader("Dataset Preview")
    if st.checkbox("Show raw data"):
        st.dataframe(data.head(100), use_container_width=True)

    if st.checkbox("Show column info"):
        st.write("**Columns:**", data.columns.tolist())
        st.write("**Data types:**")
        st.write(data.dtypes)

# ---------------------------------------------------------------------------
# Visualizations page
# ---------------------------------------------------------------------------
elif page == "Visualizations":
    st.title("Food Security Visualizations")

    data_year = data[data["Year"] == selected_year]

    # Scatterplot
    st.subheader(f"Food Insecurity by Country ({selected_year})")
    if not data_year.empty:
        fig = px.scatter(
            data_year, x="Country", y="Value", color="Value",
            title="Food Insecurity Levels by Country",
            labels={"Value": "Food Insecurity (%)"},
            color_continuous_scale="YlOrRd",
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

    # Pie chart
    st.subheader(f"Contribution to Total Food Insecurity ({selected_year})")
    top_20 = data_year.nlargest(20, "Value")
    if not top_20.empty:
        fig = px.pie(
            top_20, values="Value", names="Country",
            title="Top 20 Countries — Contribution to Food Insecurity",
            color_discrete_sequence=px.colors.sequential.YlOrRd,
        )
        st.plotly_chart(fig, use_container_width=True)

    # Top 10 bar chart
    st.subheader("Top 10 Most Food Insecure Countries")
    top_10 = data_year.nlargest(10, "Value")
    if not top_10.empty:
        fig = px.bar(
            top_10, x="Value", y="Country", orientation="h", color="Value",
            title="Countries with Highest Food Insecurity",
            color_continuous_scale="YlOrRd",
            labels={"Value": "Food Insecurity (%)"},
        )
        st.plotly_chart(fig, use_container_width=True)

    # Line chart — trend for selected country
    data_country = data[data["Country"] == selected_country]
    st.subheader(f"Food Insecurity Trend — {selected_country}")
    if not data_country.empty:
        fig = px.line(
            data_country, x="Year", y="Value",
            title=f"Food Insecurity Trend: {selected_country}",
            labels={"Value": "Food Insecurity (%)", "Year": "Year"},
        )
        st.plotly_chart(fig, use_container_width=True)

    # Histogram
    st.subheader(f"Distribution of Food Insecurity ({selected_year})")
    if not data_year.empty:
        fig = px.histogram(
            data_year, x="Value", nbins=20,
            title="Distribution of Food Insecurity Levels",
            color_discrete_sequence=["#E74C3C"],
            labels={"Value": "Food Insecurity (%)"},
        )
        st.plotly_chart(fig, use_container_width=True)

    # Box plot — regional comparison
    if "Region" in data.columns:
        st.subheader(f"Regional Comparison ({selected_year})")
        fig = px.box(
            data_year, x="Region", y="Value", color="Region",
            title="Regional Comparison of Food Insecurity",
            labels={"Value": "Food Insecurity (%)"},
        )
        st.plotly_chart(fig, use_container_width=True)

    # Heatmap
    st.subheader("Multi-Country Heatmap")
    heatmap_countries = st.multiselect(
        "Select countries:", countries, default=countries[:5]
    )
    if heatmap_countries:
        hm_data = data[data["Country"].isin(heatmap_countries)].drop_duplicates(
            subset=["Country", "Year"]
        )
        if not hm_data.empty:
            pivot = hm_data.pivot_table(
                index="Country", columns="Year", values="Value", aggfunc="mean"
            )
            fig = go.Figure(data=go.Heatmap(
                z=pivot.values, x=pivot.columns.astype(str),
                y=pivot.index, colorscale="YlOrRd",
                colorbar_title="Food Insecurity (%)",
            ))
            fig.update_layout(title="Food Insecurity Across Years")
            st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# Modeling page
# ---------------------------------------------------------------------------
elif page == "Modeling":
    st.title("Predictive Modeling")
    st.markdown("Linear Regression model predicting food insecurity values from year and country code.")

    features = ["Year", "Area Code (M49)"]
    target = "Value"

    model_data = data.dropna(subset=features + [target])
    X = model_data[features]
    y = model_data[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.subheader("Model Performance")
    col1, col2 = st.columns(2)
    col1.metric("Mean Squared Error", f"{mse:.2f}")
    col2.metric("R-Squared", f"{r2:.4f}")

    st.subheader("Predictions vs Actual")
    results = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred})
    st.dataframe(results.head(20), use_container_width=True)

    # Scatter plot of predictions
    fig = px.scatter(
        results.head(200), x="Actual", y="Predicted",
        title="Predicted vs Actual Values",
        labels={"Actual": "Actual Value", "Predicted": "Predicted Value"},
    )
    fig.add_trace(go.Scatter(
        x=[results["Actual"].min(), results["Actual"].max()],
        y=[results["Actual"].min(), results["Actual"].max()],
        mode="lines", name="Perfect Prediction",
        line=dict(dash="dash", color="red"),
    ))
    st.plotly_chart(fig, use_container_width=True)
