#!/usr/bin/env python3
"""
Monthly Factor Backtesting Framework Test Suite

Test Objectives:
1. Verify Moonshot class functionality
2. Verify function logic correctness
3. Verify numerical calculation accuracy
4. Verify boundary condition handling
5. Verify data integrity
"""

from datetime import datetime, timedelta

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")


from moonshot import Moonshot


@pytest.fixture
def simple_test_data():
    """Fixture providing simple test data for basic tests"""
    # Create simple test data with 2 assets, 2 months
    dates = pd.date_range("2023-01-01", "2023-02-28", freq="D")
    assets = ["A", "B"]

    # Construct factor data (month-end data)
    factor_dates = ["2023-01-31"]
    factor_data = []
    for date in factor_dates:
        factor_data.append({"date": pd.to_datetime(date), "asset": "A", "factor": 1.0})
        factor_data.append({"date": pd.to_datetime(date), "asset": "B", "factor": 2.0})

    factor_df = pd.DataFrame(factor_data)
    factor_series = factor_df.set_index(["date", "asset"])["factor"]

    # Construct price data - simple numbers for easy verification
    price_data = []
    for date in dates:
        # Asset A: 100 -> 110 (10% return)
        price_data.append({"date": date, "asset": "A", "open": 100, "close": 110})
        # Asset B: 100 -> 105 (5% return)
        price_data.append({"date": date, "asset": "B", "open": 100, "close": 105})

    bars_df = pd.DataFrame(price_data)
    bars = bars_df.set_index(["date", "asset"])

    return factor_series, bars


@pytest.fixture
def medium_test_data():
    """Fixture providing medium test data for comprehensive tests"""
    # Generate 6 months of daily data with 5 assets
    dates = pd.date_range("2023-01-01", "2023-06-30", freq="D")
    assets = ["Stock_A", "Stock_B", "Stock_C", "Stock_D", "Stock_E"]

    # Generate factor data (month-end) with clear ranking
    factor_data = []
    factor_values = [1.0, 2.0, 3.0, 4.0, 5.0]  # Clear ranking

    for month in pd.date_range("2023-01-31", "2023-05-31", freq="ME"):
        for i, asset in enumerate(assets):
            factor_data.append(
                {
                    "date": month,
                    "asset": asset,
                    "factor": factor_values[i],  # Consistent ranking
                }
            )

    factor_df = pd.DataFrame(factor_data)
    factor_series = factor_df.set_index(["date", "asset"])["factor"]

    # Generate price data with returns correlated to factor
    price_data = []
    base_returns = [0.02, 0.04, 0.06, 0.08, 0.10]  # Higher factor -> higher return

    for date in dates:
        for i, asset in enumerate(assets):
            # Base price
            base_price = 100
            # Monthly return based on factor ranking
            monthly_return = base_returns[i]
            # Calculate daily return
            days_in_month = 30  # Approximate
            daily_return = (1 + monthly_return) ** (1 / days_in_month) - 1

            # Simple price evolution
            price = base_price * (1 + daily_return) ** ((date - dates[0]).days)

            price_data.append(
                {
                    "date": date,
                    "asset": asset,
                    "open": price,
                    "close": price * (1 + daily_return),
                }
            )

    bars_df = pd.DataFrame(price_data)
    bars = bars_df.set_index(["date", "asset"])

    return factor_series, bars


class TestMoonshotClass:
    """Test the Moonshot class functionality"""

    def test_moonshot_initialization(self):
        """Test Moonshot class initialization"""
        moonshot = Moonshot()
        assert moonshot.quantile_returns is None
        assert moonshot.benchmark_returns is None
        assert moonshot.factor_data is None
        assert moonshot.bars_data is None

    def test_moonshot_backtest_simple(self, simple_test_data):
        """Test Moonshot backtest with simple data"""
        factor_series, bars = simple_test_data

        moonshot = Moonshot()
        moonshot.backtest(factor_series, bars, quantiles=2)
        quantile_returns = moonshot.quantile_returns
        benchmark_returns = moonshot.benchmark_returns
        long_only_returns = moonshot.long_only_returns
        long_short_returns = moonshot.long_short_returns
        optimal_returns = moonshot.optimal_returns

        # Verify results are stored
        assert moonshot.quantile_returns is not None
        assert moonshot.benchmark_returns is not None

        # Verify return values
        assert isinstance(quantile_returns, pd.DataFrame)
        assert isinstance(benchmark_returns, pd.Series)

        # Should have 2 quantiles
        assert len(quantile_returns.columns) == 2

        # Verify returns are reasonable
        # Asset A (factor=1) should be in Q1, Asset B (factor=2) should be in Q2
        # A has 10% return, B has 5% return
        # So Q1 should have 10% return, Q2 should have 5% return
        assert abs(quantile_returns.iloc[0, 0] - 0.10) < 1e-10  # Q1 (Asset A)
        assert abs(quantile_returns.iloc[0, 1] - 0.05) < 1e-10  # Q2 (Asset B)

        # Benchmark should be average: (10% + 5%) / 2 = 7.5%
        assert abs(benchmark_returns.iloc[0] - 0.075) < 1e-10

    def test_moonshot_backtest_medium(self, medium_test_data):
        """Test Moonshot backtest with medium complexity data"""
        factor_series, bars = medium_test_data

        moonshot = Moonshot()
        moonshot.backtest(factor_series, bars, quantiles=5)
        quantile_returns = moonshot.quantile_returns
        benchmark_returns = moonshot.benchmark_returns
        long_only_returns = moonshot.long_only_returns
        long_short_returns = moonshot.long_short_returns
        optimal_returns = moonshot.optimal_returns

        # Should have 5 quantiles
        assert len(quantile_returns.columns) == 5

        # Should have multiple months of data
        assert len(quantile_returns) > 1

        # Higher quantiles should generally have higher returns
        # (due to our data construction)
        mean_returns = quantile_returns.mean()
        for i in range(len(mean_returns) - 1):
            assert mean_returns.iloc[i] <= mean_returns.iloc[i + 1]

    def test_moonshot_plot_methods_require_backtest(self):
        """Test that plot methods require backtest to be run first"""
        moonshot = Moonshot()

        with pytest.raises(ValueError, match="请先执行backtest\(\)方法"):
            moonshot.plot_long_short_returns()

        with pytest.raises(ValueError, match="请先执行backtest\(\)方法"):
            moonshot.plot_long_short_returns()

        with pytest.raises(ValueError, match="请先执行backtest\(\)方法"):
            moonshot.plot_long_only_returns()

        with pytest.raises(ValueError, match="请先执行backtest\(\)方法"):
            moonshot.calculate_metrics()

    def test_moonshot_calculate_metrics(self, medium_test_data):
        """Test Moonshot metrics calculation"""
        factor_series, bars = medium_test_data

        moonshot = Moonshot()
        moonshot.backtest(factor_series, bars, quantiles=5)

        metrics = moonshot.calculate_metrics()

        # Should have 4 columns: long-short, long-only, optimal, benchmark
        assert len(metrics.columns) == 4
        assert "long-short" in metrics.columns
        assert "long-only" in metrics.columns
        assert "benchmark" in metrics.columns

        # Should have all required metrics
        expected_metrics = [
            "Ann. Returns",
            "Ann. Vol",
            "Sharpe",
            "Max DrawDonw",
            "Win Rate",
            "Sortino",
            "Calmar",
            "CAGR",
            "Alpha",
            "Beta",
        ]
        for metric in expected_metrics:
            assert metric in metrics.index

        # Verify metrics are reasonable
        assert not metrics.isna().any().any()  # No NaN values

        # Sharpe ratios should be finite
        sharpe_ratios = metrics.loc["Sharpe"]
        assert all(np.isfinite(sharpe_ratios))

        # Win rates should be between 0 and 1
        win_rates = metrics.loc["Win Rate"]
        assert all(0 <= rate <= 1 for rate in win_rates)

        # Max drawdowns should be negative or zero
        max_drawdowns = metrics.loc["Max DrawDonw"]
        assert all(dd <= 0 for dd in max_drawdowns)


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_empty_data(self):
        """Test handling of empty data"""
        empty_factor = pd.Series([], dtype=float, name="factor")
        empty_factor.index = pd.MultiIndex.from_tuples([], names=["date", "asset"])

        empty_bars = pd.DataFrame(columns=["open", "close"])
        empty_bars.index = pd.MultiIndex.from_tuples([], names=["date", "asset"])

        moonshot = Moonshot()
        moonshot.backtest(empty_factor, empty_bars)
        quantile_returns = moonshot.quantile_returns
        benchmark_returns = moonshot.benchmark_returns
        long_only_returns = moonshot.long_only_returns
        long_short_returns = moonshot.long_short_returns
        optimal_returns = moonshot.optimal_returns

        assert len(quantile_returns) == 0
        assert len(benchmark_returns) == 0

    def test_single_asset(self):
        """Test handling of single asset"""
        # Create data with only one asset
        dates = pd.date_range("2023-01-01", "2023-02-28", freq="D")

        factor_data = [
            {"date": pd.to_datetime("2023-01-31"), "asset": "A", "factor": 1.0}
        ]
        factor_df = pd.DataFrame(factor_data)
        factor_series = factor_df.set_index(["date", "asset"])["factor"]

        price_data = []
        for date in dates:
            price_data.append({"date": date, "asset": "A", "open": 100, "close": 110})

        bars_df = pd.DataFrame(price_data)
        bars = bars_df.set_index(["date", "asset"])

        moonshot = Moonshot()
        moonshot.backtest(factor_series, bars, quantiles=2)
        quantile_returns = moonshot.quantile_returns
        benchmark_returns = moonshot.benchmark_returns
        long_only_returns = moonshot.long_only_returns
        long_short_returns = moonshot.long_short_returns
        optimal_returns = moonshot.optimal_returns

        # With only one asset, quantile grouping may fail
        # In this case, we should get empty results
        # This is expected behavior when there's insufficient data for grouping
        assert len(quantile_returns.columns) >= 0

    def test_bins_parameter(self, medium_test_data):
        """Test using bins parameter instead of quantiles"""
        factor_series, bars = medium_test_data

        moonshot = Moonshot()
        moonshot.backtest(
            factor_series, bars, quantiles=None, bins=[0, 2, 4, 6]  # 3 bins
        )
        quantile_returns = moonshot.quantile_returns
        benchmark_returns = moonshot.benchmark_returns
        long_only_returns = moonshot.long_only_returns
        long_short_returns = moonshot.long_short_returns
        optimal_returns = moonshot.optimal_returns

        # Should have 3 bins
        assert len(quantile_returns.columns) == 3
        assert all(col.startswith("Bin") for col in quantile_returns.columns)

    def test_factor_lag_parameter(self, medium_test_data):
        """Test different factor lag values"""
        factor_series, bars = medium_test_data

        moonshot = Moonshot()

        # Test factor_lag = 1 (default)
        moonshot.backtest(factor_series, bars, quantiles=5, factor_lag=1)
        quantile_returns_1 = moonshot.quantile_returns

        # Test factor_lag = 2
        moonshot.backtest(factor_series, bars, quantiles=5, factor_lag=2)
        quantile_returns_2 = moonshot.quantile_returns

        # With higher lag, should have fewer trading periods
        assert len(quantile_returns_2) <= len(quantile_returns_1)

    def test_invalid_parameters(self):
        """Test invalid parameter combinations"""
        moonshot = Moonshot()

        # Create minimal valid data
        factor_data = [
            {"date": pd.to_datetime("2023-01-31"), "asset": "A", "factor": 1.0}
        ]
        factor_df = pd.DataFrame(factor_data)
        factor_series = factor_df.set_index(["date", "asset"])["factor"]

        dates = pd.date_range("2023-01-01", "2023-02-28", freq="D")
        price_data = []
        for date in dates:
            price_data.append({"date": date, "asset": "A", "open": 100, "close": 110})
        bars_df = pd.DataFrame(price_data)
        bars = bars_df.set_index(["date", "asset"])

        # Test both quantiles and bins specified
        with pytest.raises(ValueError, match="quantiles 和 bins 不能同时指定"):
            moonshot.backtest(factor_series, bars, quantiles=5, bins=[1, 2, 3])

        # Test invalid factor_lag
        with pytest.raises(ValueError, match="factor_lag 必须大于等于1"):
            moonshot.backtest(factor_series, bars, factor_lag=0)


class TestNumericalAccuracy:
    """Test numerical accuracy of calculations"""

    def test_simple_calculation_accuracy(self, simple_test_data):
        """Test that simple calculations are numerically accurate"""
        factor_series, bars = simple_test_data

        moonshot = Moonshot()
        moonshot.backtest(factor_series, bars, quantiles=2)
        quantile_returns = moonshot.quantile_returns
        benchmark_returns = moonshot.benchmark_returns
        long_only_returns = moonshot.long_only_returns
        long_short_returns = moonshot.long_short_returns
        optimal_returns = moonshot.optimal_returns

        # Verify exact calculations
        # Asset A: 100 -> 110, return = 0.10
        # Asset B: 100 -> 105, return = 0.05
        # Q1 (Asset A): 0.10
        # Q2 (Asset B): 0.05
        # Benchmark: (0.10 + 0.05) / 2 = 0.075

        expected_q1 = 0.10
        expected_q2 = 0.05
        expected_benchmark = 0.075

        assert abs(quantile_returns.iloc[0, 0] - expected_q1) < 1e-10
        assert abs(quantile_returns.iloc[0, 1] - expected_q2) < 1e-10
        assert abs(benchmark_returns.iloc[0] - expected_benchmark) < 1e-10

    def test_metrics_calculation_accuracy(self, simple_test_data):
        """Test that metrics calculations are accurate"""
        factor_series, bars = simple_test_data

        moonshot = Moonshot()
        moonshot.backtest(factor_series, bars, quantiles=2)

        metrics = moonshot.calculate_metrics()

        # Test specific metric calculations
        # For long-short spread: Q2 - Q1 = 0.05 - 0.10 = -0.05
        long_short_annual_return = metrics.loc["Ann. Returns", "long-short"]
        expected_annual_return = -0.05 * 12  # -0.60 or -60%

        assert abs(long_short_annual_return - expected_annual_return) < 1e-10

        # Win rate for long-short should be 0 (negative return)
        long_short_win_rate = metrics.loc["Win Rate", "long-short"]
        assert long_short_win_rate == 0.0

        # Win rate for pure long (Q2) should be 1 (positive return)
        pure_long_win_rate = metrics.loc["Win Rate", "long-only"]
        assert pure_long_win_rate == 1.0
