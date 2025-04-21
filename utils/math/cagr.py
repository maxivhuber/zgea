import pandas as pd

from utils.math import gmean


def cagr(df: pd.DataFrame, column: str = "sum") -> float:
    df = df.sort_index()
    yearly_results = df[column].pct_change(1, freq="YE").dropna()

    cagr = gmean(yearly_results) * 100
    return cagr
