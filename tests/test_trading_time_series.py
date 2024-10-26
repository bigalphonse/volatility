import pytest
import pandas as pd
from src.trading_time_series import TradingTimeSeries


def test_correlation():
    # Create simple series for testing
    data1 = pd.Series([1, 2, 3, 4, 5], index=pd.date_range("2023-01-01", periods=5))
    data2 = pd.Series([5, 4, 3, 2, 1], index=pd.date_range("2023-01-01", periods=5))

    series1 = TradingTimeSeries(data=data1)
    series2 = TradingTimeSeries(data=data2)

    # Test correlation
    correlation = series1.compute_correlation(series2)
    assert correlation == pytest.approx(-1, abs=1e-9)


def test_mutual_information():
    # Similar setup for mutual information testing
    pass
