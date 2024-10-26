from dataclasses import dataclass, field
import pandas as pd
import yfinance as yf
import numpy as np
from sklearn.metrics import mutual_info_score


@dataclass
class TradingTimeSeries:
    """
    Data class for handling financial trading time series.
    """

    data: pd.Series = field(default_factory=pd.Series)

    def __post_init__(self):
        if not isinstance(self.data.index, pd.DatetimeIndex):
            raise ValueError("Data must be indexed by a DateTimeIndex")

    def align_with(self, other: "TradingTimeSeries") -> pd.DataFrame:
        """
        Aligns the current series with another series by date.
        """
        aligned_data = pd.concat([self.data, other.data], axis=1, join="inner")
        aligned_data.columns = ["Series1", "Series2"]
        return aligned_data

    def compute_correlation(self, other: "TradingTimeSeries") -> float:
        """
        Computes the correlation with another time series.
        """
        aligned_data = self.align_with(other)
        correlation = aligned_data["Series1"].corr(aligned_data["Series2"])
        return correlation

    def compute_mutual_information(
        self, other: "TradingTimeSeries", bins: int = 10
    ) -> float:
        """
        Computes the mutual information with another time series.
        """
        aligned_data = self.align_with(other)
        series1_binned = pd.cut(aligned_data["Series1"], bins=bins, labels=False)
        series2_binned = pd.cut(aligned_data["Series2"], bins=bins, labels=False)
        mutual_info = mutual_info_score(series1_binned, series2_binned)
        return mutual_info

    def resample(self, rule: str) -> "TradingTimeSeries":
        """
        Resamples the series based on the given frequency rule (e.g., 'D', 'W', 'M').
        """
        resampled_data = self.data.resample(rule).mean()
        return TradingTimeSeries(data=resampled_data)

    def fetch_vix_series(self, vix_type: str = "regular") -> "TradingTimeSeries":
        """
        Fetches the specified VIX series from Yahoo Finance over the time range of the current series.

        Args:
            vix_type (str): Type of VIX to fetch - 'regular', 'vix9d', or 'vix1d'.

        Returns:
            A new instance of TradingTimeSeries containing the VIX series data.
        """
        # Map VIX types to Yahoo Finance tickers
        vix_tickers = {"regular": "^VIX", "vix9d": "^VIX9D", "vix1d": "^VIX1D"}

        if vix_type not in vix_tickers:
            raise ValueError(
                "Invalid vix_type. Choose from 'regular', 'vix9d', or 'vix1d'."
            )

        start_date = self.data.index.min().strftime("%Y-%m-%d")
        end_date = self.data.index.max().strftime("%Y-%m-%d")
        ticker = vix_tickers[vix_type]

        # Fetch the VIX data from Yahoo Finance
        vix_data = yf.download(ticker, start=start_date, end=end_date, progress=False)

        # Ensure VIX data is indexed by date
        vix_series = vix_data["Close"]
        vix_series.index = pd.to_datetime(vix_series.index)

        return TradingTimeSeries(data=vix_series)

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return f"TradingTimeSeries(length={len(self)}, start={self.data.index.min()}, end={self.data.index.max()})"
