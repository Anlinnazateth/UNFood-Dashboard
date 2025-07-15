import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Streamlit app configuration
st.set_page_config(page_title="Global Food Security Dashboard", layout="wide")

# App title
st.title("Global Food Security Dashboard")

# Load the dataset
data = pd.read_csv('food_security.csv')

# Sidebar options
st.sidebar.title("Navigation")
selected_option = st.sidebar.radio("Choose an option", ["Visualization", "Modeling"])

# Common Data Preprocessing
def preprocess_data(data):
    required_columns = ['Year', 'Value', 'Country', 'Area Code (M49)']
    missing_columns = [col for col in required_columns if col not in data.columns]

    if missing_columns:
        st.error(f"The dataset is missing the following required columns: {missing_columns}")
        st.stop()

    data['Year'] = data['Year'].str.split('-').apply(lambda x: int(x[0]) + 1)
    data['Value'] = data['Value'].str.replace('<', '').str.replace('>', '').astype(float)
    data = data.drop_duplicates()
    return data

data = preprocess_data(data)

if selected_option == "Visualization":
    st.header("Data Visualization")

    # Display dataset information
    if st.sidebar.checkbox("Show Dataset Overview"):
        st.subheader("Dataset Overview")
        st.write(data.head())
        st.write("Columns:", data.columns.tolist())
        st.write("Data Types:", data.dtypes)

    # Sidebar filters
    years = sorted(data['Year'].unique())
    countries = sorted(data['Country'].unique())
    selected_year = st.sidebar.selectbox("Select Year", years)
    selected_country = st.sidebar.selectbox("Select Country", countries)
    selected_countries = st.sidebar.multiselect("Select Countries for Heatmap", countries, default=countries[:5])

    # Filter data by selected year
    data_year = data[data['Year'] == selected_year]

    # Visualizations
    st.subheader(f"Scatterplot of Food Insecurity Levels ({selected_year})")
    scatterplot = px.scatter(
        data_year, x="Country", y="Value", color="Value",
        title="Scatterplot of Food Insecurity by Country",
        labels={"Value": "Food Insecurity (%)", "Country": "Country"},
        color_continuous_scale="YlOrRd"
    )
    st.plotly_chart(scatterplot, use_container_width=True)

    st.subheader(f"Pie Chart of Food Insecurity Contribution ({selected_year})")
    pie_chart = px.pie(
        data_year, values="Value", names="Country",
        title="Percentage Contribution to Total Food Insecurity",
        color_discrete_sequence=px.colors.sequential.YlOrRd
    )
    st.plotly_chart(pie_chart, use_container_width=True)

    st.subheader("Top Countries with Severe Food Insecurity")
    bar_chart_data = data_year.nlargest(10, 'Value')
    bar_chart = px.bar(
        bar_chart_data, x="Value", y="Country", orientation="h", color="Value",
        title="Countries with Highest Food Insecurity",
        color_continuous_scale="YlOrRd",
        labels={"Value": "Food Insecurity (%)"}
    )
    st.plotly_chart(bar_chart, use_container_width=True)

    st.subheader(f"Food Insecurity Trend for {selected_country}")
    data_country = data[data['Country'] == selected_country]
    line_chart = px.line(
        data_country, x="Year", y="Value",
        title=f"Food Insecurity Trend: {selected_country}",
        labels={"Value": "Food Insecurity (%)", "Year": "Year"}
    )
    st.plotly_chart(line_chart, use_container_width=True)

    st.subheader(f"Histogram of Food Insecurity Levels ({selected_year})")
    histogram = px.histogram(
        data_year, x="Value", nbins=20,
        title="Distribution of Food Insecurity Levels",
        color_discrete_sequence=["red"],
        labels={"Value": "Food Insecurity (%)"}
    )
    st.plotly_chart(histogram, use_container_width=True)

    if 'Region' in data.columns:
        st.subheader(f"Food Insecurity Across Regions ({selected_year})")
        box_plot = px.box(
            data_year, x="Region", y="Value", color="Region",
            title="Regional Comparison of Food Insecurity",
            labels={"Value": "Food Insecurity (%)"},
            color_discrete_sequence=px.colors.sequential.YlOrRd
        )
        st.plotly_chart(box_plot, use_container_width=True)

    st.subheader("Heatmap of Food Insecurity Levels")
    data_heatmap = data[data['Country'].isin(selected_countries)].drop_duplicates(subset=['Country', 'Year'])
    data_pivot = data_heatmap.pivot(index="Country", columns="Year", values="Value")
    heatmap = go.Figure(data=go.Heatmap(
        z=data_pivot.values, x=data_pivot.columns, y=data_pivot.index,
        colorscale="YlOrRd", colorbar_title="Food Insecurity (%)"
    ))
    heatmap.update_layout(title="Food Insecurity Levels Across Years")
    st.plotly_chart(heatmap, use_container_width=True)

elif selected_option == "Modeling":
    st.header("Data Modeling")

    # Feature selection and preprocessing
    features = ['Year', 'Area Code (M49)']
    target = 'Value'

    # Handle missing values
    if data[target].isna().sum() > 0:
        data = data.dropna(subset=[target])

    X = data[features]
    y = data[target]

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Model evaluation
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.subheader("Model Performance")
    st.write(f"Mean Squared Error (MSE): {mse:.2f}")
    st.write(f"R-squared (R2): {r2:.2f}")

    # Display predictions
    predictions = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred})
    st.subheader("Sample Predictions")
    st.write(predictions.head(10))
