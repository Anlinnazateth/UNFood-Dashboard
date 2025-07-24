# Global Food Security Dashboard

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An interactive dashboard for visualizing global food insecurity data from the UN FAO. Explore trends by year and country through maps, charts, and ML models.

## Features

### Overview
- Key metrics: total countries, year range, data points
- Dataset preview and column information

### Visualizations
- **Scatterplot** — Food insecurity levels by country
- **Pie Chart** — Top 20 countries' contribution to total insecurity
- **Bar Chart** — Top 10 most food-insecure countries
- **Line Chart** — Trend over time for a selected country
- **Histogram** — Distribution of insecurity levels
- **Box Plot** — Regional comparison (if Region data available)
- **Heatmap** — Multi-country comparison across years

### ML Modeling
- Linear Regression model predicting food insecurity values
- MSE and R-squared metrics display
- Interactive predicted vs actual scatter plot

## Tech Stack

- **Frontend:** Streamlit, Plotly
- **Data:** Pandas, NumPy
- **ML:** scikit-learn

## Installation

```bash
git clone https://github.com/Anlinnazateth/UNFood-Dashboard.git
cd UNFood-Dashboard
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. Use the sidebar to navigate between Overview, Visualizations, and Modeling sections.

## Data Source

**UN FAO Suite of Food Security Indicators**

The `food_security.csv` dataset contains ~18,700 rows with 15 columns:

| Column | Description |
|--------|-------------|
| `Country` | Country name |
| `Area Code (M49)` | UN M49 numeric country code |
| `Year` | Year range (e.g., "2000-2002") |
| `Value` | Food insecurity value (% or millions) |
| `Item` | Indicator description |
| `Unit` | Measurement unit (% or million No) |
| `Flag` | Estimation type (E=Estimated, A=Official) |

Coverage: 2000–2023, spanning multiple food security indicators.

## Project Structure

```
UNFood-Dashboard/
├── app.py              # Main Streamlit app (consolidated)
├── food_security.csv   # UN FAO dataset
├── requirements.txt    # Python dependencies
├── LICENSE
├── .gitignore
├── .streamlit/
│   └── config.toml     # Theme configuration
├── .github/
│   └── workflows/
│       └── ci.yml      # CI pipeline
└── tests/
    └── test_app.py     # Unit tests
```

## Running Tests

```bash
pip install pytest
pytest tests/ -v
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Open a Pull Request

## License

MIT License. See [LICENSE](LICENSE) for details.
