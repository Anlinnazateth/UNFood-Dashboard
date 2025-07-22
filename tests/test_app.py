"""Tests for the UNFood Dashboard."""

import os
import pandas as pd
import numpy as np
import pytest

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "food_security.csv")


@pytest.fixture
def raw_data():
    return pd.read_csv(DATA_PATH, encoding="utf-8-sig")


@pytest.fixture
def processed_data(raw_data):
    data = raw_data.copy()
    data["Year"] = (
        data["Year"]
        .astype(str)
        .str.split("-")
        .apply(lambda x: int(x[0]) + 1 if len(x) == 2 else int(x[0]))
    )
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


class TestDataLoading:
    def test_csv_loads(self, raw_data):
        assert not raw_data.empty

    def test_required_columns(self, raw_data):
        for col in ["Year", "Value", "Country", "Area Code (M49)"]:
            assert col in raw_data.columns

    def test_has_rows(self, raw_data):
        assert len(raw_data) > 1000


class TestPreprocessing:
    def test_year_is_numeric(self, processed_data):
        assert pd.api.types.is_numeric_dtype(processed_data["Year"])

    def test_year_reasonable_range(self, processed_data):
        assert processed_data["Year"].min() >= 1990
        assert processed_data["Year"].max() <= 2030

    def test_value_is_numeric(self, processed_data):
        assert pd.api.types.is_numeric_dtype(processed_data["Value"])

    def test_no_null_values(self, processed_data):
        assert processed_data["Value"].isna().sum() == 0

    def test_duplicates_removed(self, processed_data):
        assert len(processed_data) == len(processed_data.drop_duplicates())


class TestFiltering:
    def test_filter_by_year(self, processed_data):
        year = processed_data["Year"].iloc[0]
        filtered = processed_data[processed_data["Year"] == year]
        assert all(filtered["Year"] == year)

    def test_filter_by_country(self, processed_data):
        country = processed_data["Country"].iloc[0]
        filtered = processed_data[processed_data["Country"] == country]
        assert all(filtered["Country"] == country)

    def test_top_n(self, processed_data):
        top10 = processed_data.nlargest(10, "Value")
        assert len(top10) == 10
