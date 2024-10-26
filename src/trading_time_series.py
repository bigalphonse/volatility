from dataclasses import dataclass, field
import pandas as pd
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

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        return f"TradingTimeSeries(length={len(self)}, start={self.data.index.min()}, end={self.data.index.max()})"
