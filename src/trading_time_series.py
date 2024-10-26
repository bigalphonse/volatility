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

    def vix_futures_term_structure(self, date: str) -> pd.Series:
        """
        Computes the VIX futures monthly term structure at a given date.
        """
        vix_tickers = ["VX1", "VX2", "VX3", "VX4", "VX5", "VX6", "VX7", "VX8"]
        vix_futures_data = {}

        for ticker in vix_tickers:
            vix_future = yf.download(
                f"{ticker}=F", start=date, end=date, progress=False
            )
            if not vix_future.empty:
                vix_futures_data[ticker] = vix_future["Close"].iloc[0]

        term_structure = pd.Series(vix_futures_data)
        term_structure.index = [f"Month {i+1}" for i in range(len(term_structure))]
        return term_structure

    def term_structure_type(self, term_structure: pd.Series) -> str:
        """
        Determines if the VIX futures term structure is in contango, backwardation, or undefined.
        """
        if len(term_structure) < 2:
            return "undefined"
        if term_structure.iloc[0] < term_structure.iloc[-1]:
            return "contango"
        elif term_structure.iloc[0] > term_structure.iloc[-1]:
            return "backwardation"
        else:
            return "undefined"

    def generate_vix_term_structure_series(self) -> "TradingTimeSeries":
        """
        Generates the VIX term structure and type for each date in the series.
        Returns a new TradingTimeSeries with term structure types.
        """
        term_structure_results = {}

        for date in self.data.index:
            date_str = date.strftime("%Y-%m-%d")
            term_structure = self.vix_futures_term_structure(date_str)
            structure_type = self.term_structure_type(term_structure)
            term_structure_results[date] = structure_type

        # Create a new series with the term structure type for each date
        term_structure_series = pd.Series(term_structure_results, index=self.data.index)
        return TradingTimeSeries(data=term_structure_series)

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return f"TradingTimeSeries(length={len(self)}, start={self.data.index.min()}, end={self.data.index.max()})"
