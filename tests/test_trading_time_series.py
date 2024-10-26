import pytest
import pandas as pd
from src.trading_time_series import TradingTimeSeries
import socket


# Attempts to establish a socket connection to Google's DNS server (8.8.8.8) on port 53, which is a reliable check for internet connectivity.


def is_internet_accessible(host="8.8.8.8", port=53, timeout=3):
    """
    Checks for internet connectivity by attempting to connect to a known host.
    """
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except socket.error:
        return False


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


# @pytest.mark.skipif(not is_internet_accessible(), reason="No internet connection")
def test_fetch_vix_series():
    # Create a sample time series with a defined date range
    data = pd.Series(
        [100, 101, 102, 103, 104], index=pd.date_range("2023-01-01", periods=5)
    )
    series = TradingTimeSeries(data=data)

    # Fetch the VIX series over the same date range
    vix_series = series.fetch_vix_series()

    # Ensure the VIX series is a TradingTimeSeries instance
    assert isinstance(vix_series, TradingTimeSeries)

    # Check that the VIX series is not empty
    assert not vix_series.data.empty

    # Ensure that the VIX series has the same start and end dates as the original series
    assert vix_series.data.index.min() >= series.data.index.min()
    assert vix_series.data.index.max() <= series.data.index.max()
