from unittest.mock import patch

import matplotlib
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_index_equal, assert_series_equal

matplotlib.use("Agg")

import moonshot
from moonshot import Moonshot


@pytest.fixture
def simple_test_data():
    dates = pd.date_range("2023-01-01", "2023-02-28", freq="D")

    # Construct factor data (month-end data)
    factor_dates = ["2023-01-31"]
    factor_data = []
    for date in factor_dates:
        factor_data.append({"date": pd.to_datetime(date), "asset": "A", "factor": 1.0})
        factor_data.append({"date": pd.to_datetime(date), "asset": "B", "factor": 2.0})

    factor_df = pd.DataFrame(factor_data)

    # Construct price data - simple numbers for easy verification
    price_data = []
    for date in dates:
        # Asset A: 100 -> 110 (10% return)
        price_data.append({"date": date, "asset": "A", "open": 100, "close": 110})
        # Asset B: 100 -> 105 (5% return)
        price_data.append({"date": date, "asset": "B", "open": 100, "close": 105})

    bars = pd.DataFrame(price_data)

    return factor_df, bars


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

    bars = pd.DataFrame(price_data)

    return factor_df, bars


@pytest.fixture
def mnshot(simple_test_data) -> Moonshot:
    _, bars = simple_test_data
    return Moonshot(bars)


class TestMoonshotClass:
    def test_check_factor_type(self, mnshot):
        assert "discrete" == mnshot.check_factor_type(pd.Series([1, 0, -1]))
        assert "discrete" == mnshot.check_factor_type(pd.Series([1, 1, 1]))
        assert "discrete" == mnshot.check_factor_type(pd.Series([1.0, 0.0, -1.0]))

        assert "continuous" == mnshot.check_factor_type(
            pd.Series([1, 2, 3], dtype=np.float32)
        )

    def test_is_month_indexed(self, mnshot):
        assert mnshot.is_month_indexed(mnshot.data)

        df = pd.DataFrame()
        assert not mnshot.is_month_indexed(df)

    def test_append_factor(self, simple_test_data):
        factors, bars = simple_test_data
        ms = Moonshot(bars)

        with pytest.raises(ValueError) as e:
            ms.append_factor(factors, "A")

        assert str(e.value) == "因子数据中不存在列: A"

        with pytest.raises(ValueError) as e:
            ms.append_factor(factors, "factor", resample="first")
            ms.append_factor(factors, "factor", resample="first")

        assert str(e.value) == "重复加入因子：factor"

        ms = Moonshot(bars)
        factors["factor"] = [1, 0]
        with patch.object(moonshot.moonshot.logger, "warning") as mock_warning:
            ms.append_factor(factors, "factor", 5, resample="first")
            assert mock_warning.called

        ms = Moonshot(bars)
        with pytest.raises(ValueError) as e:
            ms.append_factor(factors, "factor")

        assert str(e.value) == "数据未按月份索引，请使用resample_method参数指定重采样方法"

        ms.append_factor(factors, "factor", resample="first")
        result = ms.data["factor"].values.tolist()
        expected = [1.0, 0.0, np.nan, np.nan]

        np.testing.assert_array_equal(result, expected)

    def test_calculate_benchmark_returns(self):
        """Test _calculate_benchmark_returns method with simple data for easy mental verification"""
        # Create simple test data with 2 assets and 3 months
        # Asset A: open=100, close=110 in month1 (10% return)
        # Asset A: close=121 in month2 (10% return from 110)
        # Asset A: close=133.1 in month3 (10% return from 121)
        # Asset B: open=200, close=210 in month1 (5% return)
        # Asset B: close=220.5 in month2 (5% return from 210)
        # Asset B: close=231.525 in month3 (5% return from 220.5)

        dates = pd.date_range("2023-01-01", "2023-03-31", freq="D")

        price_data = []
        for date in dates:
            month = date.month
            if month == 1:
                # Month 1
                price_data.append(
                    {"date": date, "asset": "A", "open": 100, "close": 110}
                )
                price_data.append(
                    {"date": date, "asset": "B", "open": 200, "close": 210}
                )
            elif month == 2:
                # Month 2
                price_data.append(
                    {"date": date, "asset": "A", "open": 110, "close": 121}
                )
                price_data.append(
                    {"date": date, "asset": "B", "open": 210, "close": 220.5}
                )
            else:
                # Month 3
                price_data.append(
                    {"date": date, "asset": "A", "open": 121, "close": 133.1}
                )
                price_data.append(
                    {"date": date, "asset": "B", "open": 220.5, "close": 231.525}
                )

        bars = pd.DataFrame(price_data)
        ms = Moonshot(bars)

        # Calculate benchmark returns
        benchmark_returns = ms._calculate_benchmark_returns()

        # Expected values:
        # Month 1: (110/100-1 + 210/200-1) / 2 = (0.1 + 0.05) / 2 = 0.075 (7.5%)
        # Month 2: (121/110-1 + 220.5/210-1) / 2 = (0.1 + 0.05) / 2 = 0.075 (7.5%)
        # Month 3: (133.1/121-1 + 231.525/220.5-1) / 2 = (0.1 + 0.05) / 2 = 0.075 (7.5%)
        expected_returns = pd.Series(
            [0.075, 0.075, 0.075], index=pd.period_range("2023-01", "2023-03", freq="M")
        )

        # Verify the results
        assert_series_equal(benchmark_returns, expected_returns, check_dtype=False)

    def test_calculate_benchmark_returns_with_different_returns(self):
        """Test _calculate_benchmark_returns with assets having different returns"""
        # Create test data with 3 assets and 2 months with different returns
        # Asset A: open=100, close=110 in month1 (10% return), close=132 in month2 (20% return)
        # Asset B: open=200, close=210 in month1 (5% return), close=231 in month2 (10% return)
        # Asset C: open=300, close=315 in month1 (5% return), close=346.5 in month2 (10% return)

        dates = pd.date_range("2023-01-01", "2023-02-28", freq="D")

        price_data = []
        for date in dates:
            month = date.month
            if month == 1:
                # Month 1
                price_data.append(
                    {"date": date, "asset": "A", "open": 100, "close": 110}
                )
                price_data.append(
                    {"date": date, "asset": "B", "open": 200, "close": 210}
                )
                price_data.append(
                    {"date": date, "asset": "C", "open": 300, "close": 315}
                )
            else:
                # Month 2
                price_data.append(
                    {"date": date, "asset": "A", "open": 110, "close": 132}
                )
                price_data.append(
                    {"date": date, "asset": "B", "open": 210, "close": 231}
                )
                price_data.append(
                    {"date": date, "asset": "C", "open": 315, "close": 346.5}
                )

        bars = pd.DataFrame(price_data)
        ms = Moonshot(bars)

        # Calculate benchmark returns
        benchmark_returns = ms._calculate_benchmark_returns()

        # Expected values:
        # Month 1: (110/100-1 + 210/200-1 + 315/300-1) / 3 = (0.1 + 0.05 + 0.05) / 3 = 0.066666...
        # Month 2: (132/110-1 + 231/210-1 + 346.5/315-1) / 3 = (0.2 + 0.1 + 0.1) / 3 = 0.133333...
        expected_returns = pd.Series(
            [0.06666666666666667, 0.13333333333333333],
            index=pd.period_range("2023-01", "2023-02", freq="M"),
        )

        # Verify the results
        assert_series_equal(benchmark_returns, expected_returns, check_dtype=False)

    def test_discretize_continuous_factors(self, medium_test_data):
        """Test _discretize_continuous_factors method"""
        factor, bars = medium_test_data
        ms = Moonshot(bars)

        # 增加一个新的资产
        data = ms.data.xs("Stock_A", level="asset").copy()
        data["asset"] = "Stock_F"
        data = data.set_index([data.index, "asset"])
        ms.data = pd.concat([ms.data, data]).sort_index()

        data = np.random.choice([1, 0, -1], size=36)
        factor = pd.DataFrame(data, columns=["factor_A"], index=ms.data.index)
        ms.append_factor(factor, "factor_A")
        count_A_1 = factor.query("factor_A == 1").shape[0]

        data = np.random.normal(size=36)
        top = np.percentile(data, 80)
        factor = pd.DataFrame(data, columns=["factor_B"], index=ms.data.index)
        ms.append_factor(factor, "factor_B")
        count_B_1 = factor.query("factor_B > @top").shape[0]

        data = np.random.choice([1, 0, -1], size=36)
        factor = pd.DataFrame(data, columns=["factor_C"], index=ms.data.index)
        ms.append_factor(factor, "factor_C")

        data = np.random.normal(size=36)
        factor = pd.DataFrame(data, columns=["factor_D"], index=ms.data.index)
        ms.append_factor(factor, "factor_D", quantiles=3)

        ms._discretize_continuous_factors()
        assert ms.factor_names == ["factor_A", "factor_B", "factor_C", "factor_D"]

        # 离散型保持不变
        assert len(ms.data.query("factor_A == 1")) == count_A_1
        assert len(ms.data.query("factor_B == 1")) >= len(ms.data) / 6
        assert len(ms.data.query("factor_B == -1")) >= len(ms.data) / 6

        assert len(ms.data.query("factor_D == 1")) == 12
        assert len(ms.data.query("factor_D == -1")) == 12

    def test_merge_factors_to_flag(self, mnshot):
        """Test merge_factors_to_flag method with simple data"""
        factors = mnshot.data.copy()

        factors["factor_1"] = [1, 0, -1, -1]
        factors["factor_2"] = [1, 0, 0, -1]
        factors["factor_3"] = [1, 1, 0, -1]

        mnshot.append_factor(factors, "factor_1")
        mnshot.append_factor(factors, "factor_2")
        mnshot.append_factor(factors, "factor_3")

        flags = mnshot._merge_factors_to_flag()
        assert flags.values.tolist() == [1, 0, 0, -1]

    def test_calculate_flag_returns(self, medium_test_data):
        factors, daily_bars = medium_test_data
        ms = Moonshot(daily_bars)

        factors["a"] = [
            1,
            1,
            1,
            -1,
            -1,
            1,
            1,
            0,
            -1,
            -1,
            1,
            1,
            0,
            -1,
            -1,
            1,
            0,
            0,
            -1,
            1,
            1,
            0,
            1,
            -1,
            1,
        ]

        ms.append_factor(factors, "a", resample="first")
        actual = ms._calculate_flag_returns(ms.data["a"], True)
        expect = pd.Series(
            [0, 0.037, 0.031, 0.03, 0.062, 0.06],
            index=pd.period_range("2023-01", "2023-06", freq="M"),
        )

        assert_series_equal(actual, expect, check_dtype=False, atol=1e-3)

        actual = ms._calculate_flag_returns(ms.data["a"], False)
        expect = pd.Series(
            [0, -0.023, -0.031, -0.03, -0.01, -0.01],
            index=pd.period_range("2023-01", "2023-06", freq="M"),
        )

        assert_series_equal(actual, expect, check_dtype=False, atol=1e-3)

    def test_calculate_quantile_returns(self, medium_test_data):
        factor, bars = medium_test_data
        ms = Moonshot(bars)
        ms.append_factor(factor, "factor", resample="last")

        # 01 Test with long_only=True (should only use top quantile)
        actual = ms._calculate_quantile_returns(long_only=True)
        expect = ms.data.xs("Stock_E", level="asset")["close"].pct_change()
        expect.iloc[0] = 0
        assert_series_equal(actual, expect, check_dtype=False, check_names=False)

        actual = ms.quantile_returns
        expect = pd.DataFrame(
            [
                (0, 0, 0, 0, 0),
                (0.018654, 0.037284, 0.055890, 0.074473, 0.093033),
                (0.020674, 0.041361, 0.062061, 0.082774, 0.103500),
                (0.020000, 0.040000, 0.060000, 0.080000, 0.100000),
                (0.020674, 0.041361, 0.062061, 0.082774, 0.103500),
                (0.020000, 0.040000, 0.060000, 0.080000, 0.100000),
            ]
        )

        assert_frame_equal(
            actual.reset_index(drop=True),
            expect.reset_index(drop=True),
            check_dtype=False,
            check_names=False,
            atol=1e-3,
        )

        # 02 Test with long_only=False (should use top quantile minus bottom quantile)
        actual = ms._calculate_quantile_returns(long_only=False)
        expect = (ms.quantile_returns.iloc[:, -1] - ms.quantile_returns.iloc[:, 0]) / 2

        assert_series_equal(
            actual, expect, check_dtype=False, check_names=False, check_index=False
        )

        actual = actual.index
        expect = pd.period_range("2023-01", "2023-06", freq="M")
        assert_index_equal(actual, expect, check_names=False)

    def test_run_without_factor(self, simple_test_data):
        """Test run method without adding any factor (should raise ValueError)"""
        _, bars = simple_test_data
        ms = Moonshot(bars)

        # Should raise ValueError when no factor is added
        with pytest.raises(ValueError) as excinfo:
            ms.run()

        assert "请先使用 append_factor 方法添加因子数据" in str(excinfo.value)

    def test_run_with_single_discrete_factor(self, simple_test_data):
        """Test run method with a single discrete factor"""
        factors, bars = simple_test_data
        ms = Moonshot(bars)

        # Add discrete factor
        factors["factor"] = [1, -1]  # Asset A: long, Asset B: short
        ms.append_factor(factors, "factor", resample="first")

        # Run with long_only=True
        actual_strategy_ret, _ = ms.run(long_only=True)
        stock_A = ms.data.xs("A", level="asset")
        coo = stock_A["close"] / stock_A["open"] - 1
        expect_strategy_ret = [0, coo.iloc[1]]
        np.testing.assert_array_almost_equal(
            actual_strategy_ret, expect_strategy_ret, decimal=3
        )

        # Run with long_only=False
        actual_strategy_ret, _ = ms.run(long_only=False)
        stock_B = ms.data.xs("B", level="asset")
        cooB = stock_B["close"] / stock_B["open"] - 1
        expect_strategy_ret = [0, (coo.iloc[1] - cooB.iloc[1]) / 2]
        np.testing.assert_array_almost_equal(
            actual_strategy_ret, expect_strategy_ret, decimal=3
        )

    def test_run_with_single_continuous_factor(self, medium_test_data):
        """Test run method with a single continuous factor"""
        factors, bars = medium_test_data
        ms = Moonshot(bars)

        # Add continuous factor
        ms.append_factor(factors, "factor", resample="last")

        # Run with long_only=True
        strategy_returns, benchmark_returns = ms.run(long_only=True)

        expected_strategy_ret = (ms.data["close"] / ms.data["open"] - 1).xs(
            "Stock_E", level="asset"
        )
        expected_strategy_ret.iloc[0] = 0
        expected_benchmark_ret = (
            (ms.data["close"] / ms.data["open"] - 1).unstack(0).mean()
        )
        assert_series_equal(
            strategy_returns, expected_strategy_ret, check_names=False, atol=1e-3
        )
        assert_series_equal(
            benchmark_returns, expected_benchmark_ret, check_names=False
        )

        # 保证调用时类型正确
        alpha = ms.alpha(strategy_returns, benchmark_returns)
        beta = ms.beta(strategy_returns, benchmark_returns)
        assert abs(alpha - 4.1998) < 1e-3
        assert abs(beta + 4.4184) < 1e-3
        # Run with long_only=False
        strategy_returns_ls, benchmark_returns_ls = ms.run(long_only=False)

        tmp = (ms.data["close"] / ms.data["open"] - 1).unstack(1)
        expected_strategy_ret = (tmp.loc[:, "Stock_E"] - tmp.loc[:, "Stock_A"]) / 2
        expected_strategy_ret.iloc[0] = 0
        assert_series_equal(strategy_returns_ls, expected_strategy_ret)
        assert_series_equal(
            benchmark_returns_ls, expected_benchmark_ret, check_names=False
        )

    def test_run_with_multiple_factors(self, medium_test_data):
        """Test run method with multiple factors"""
        factors, bars = medium_test_data
        factors["factor2"] = 6 - factors["factor"]
        factors["factor3"] = [1] * len(factors)

        ms = Moonshot(bars)
        for col in ("factor", "factor2", "factor3"):
            ms.append_factor(factors, col, resample="last")

        # Run with long_only=True
        strategy_returns, _ = ms.run(long_only=True)

        # 由于factor 与 factor2正好相反，所以没有一个 asset 会得到全1
        np.testing.assert_array_equal(strategy_returns.values, [0] * 6)

        # 让stock_E, stock_D 会得到全1，其余为0
        factors["factor4"] = 1
        mask = factors["asset"].isin(["Stock_D", "Stock_E"])
        factors.loc[mask, "factor4"] = 2

        ms = Moonshot(bars)
        for col in ("factor", "factor3", "factor4"):
            ms.append_factor(factors, col, resample="last")

        # Run with long_only=False
        strategy_returns_ls, benchmark_returns_ls = ms.run(long_only=False)

        # Verify returns are calculated
        assert isinstance(strategy_returns_ls, pd.Series)
        assert isinstance(benchmark_returns_ls, pd.Series)
        assert len(strategy_returns_ls) == 6

    def test_run_return_values(self, simple_test_data):
        """Test that run method returns correct strategy and benchmark returns"""
        factors, bars = simple_test_data
        ms = Moonshot(bars)

        # Add discrete factor
        factors["factor"] = [1, -1]  # Asset A: long, Asset B: short
        ms.append_factor(factors, "factor", resample="first")

        # Run with long_only=True
        strategy_returns, benchmark_returns = ms.run(long_only=True)

        # For long_only=True, strategy should only go long on Asset A (10% return)
        # Benchmark should be average of both assets (7.5% return)
        expected_strategy = pd.Series(
            [0, 0.1], index=pd.period_range("2023-01", "2023-02", freq="M")
        )
        expected_benchmark = pd.Series(
            [0.075, 0], index=pd.period_range("2023-01", "2023-02", freq="M")
        )

        assert_series_equal(strategy_returns, expected_strategy, check_dtype=False)
        assert_series_equal(benchmark_returns, expected_benchmark, check_dtype=False)

        # Run with long_only=False
        strategy_returns, benchmark_returns = ms.run(long_only=False)

        # For long_only=False, strategy should go long on Asset A (10% return) and short on Asset B (5% return)
        # Expected strategy return: (10% - 5%)/2 = 2.5%
        # Benchmark should still be average of both assets (7.5% return)
        expected_strategy_ls = pd.Series(
            [0, 0.025], index=pd.period_range("2023-01", "2023-02", freq="M")
        )

        assert_series_equal(strategy_returns, expected_strategy_ls, check_dtype=False)
        assert_series_equal(benchmark_returns, expected_benchmark, check_dtype=False)

    def test_alpha_beta(self, mnshot):
        strategy_returns = pd.Series(
            [0.05, 0.06, 0.07, 0.08, 0.09, 0.1],
            index=pd.period_range("2023-01", "2023-06", freq="M"),
        )

        benchmark_returns = pd.Series(
            [0.04, 0.05, 0.06, 0.07, 0.08, 0.09],
            index=pd.period_range("2023-01", "2023-06", freq="M"),
        )

        alpha = mnshot.alpha(strategy_returns, strategy_returns)
        assert alpha == 0

        beta = mnshot.beta(strategy_returns, benchmark_returns)
        assert beta == 1

        alpha = mnshot.alpha(strategy_returns, benchmark_returns)
        beta = mnshot.beta(strategy_returns, benchmark_returns)

        assert abs(alpha - 0.12) < 1e-3
        assert abs(beta - 1) < 1e-3

    def test_get_core_metrics(self, mnshot):
        strategy_returns = pd.Series(
            [0.05, 0.06, 0.07, 0.08, 0.09, 0.1],
            index=pd.period_range("2023-01", "2023-06", freq="M"),
        )

        benchmark_returns = pd.Series(
            [0.04, 0.05, 0.06, 0.07, 0.08, 0.09],
            index=pd.period_range("2023-01", "2023-06", freq="M"),
        )

        metrics = mnshot.get_core_metrics(strategy_returns, benchmark_returns)
        assert abs(metrics["alpha"] - 0.12) < 1e-3
        assert metrics["beta"] == 1

    def test_report(self, mnshot):
        strategy_returns = pd.Series(
            [0.05, 0.06, 0.07, 0.08, 0.09, 0.1],
            index=pd.period_range("2023-01", "2023-06", freq="M"),
        )

        benchmark_returns = pd.Series(
            [0.04, 0.05, 0.06, 0.07, 0.08, 0.09],
            index=pd.period_range("2023-01", "2023-06", freq="M"),
        )

        report = mnshot.report(
            strategy_returns, benchmark_returns, kind="metrics", display=False
        )
        assert isinstance(report, pd.DataFrame)
        actual = report.to_dict()
        assert "Start Period" in actual["Strategy"]
        assert "End Period" in actual["Strategy"]
        assert actual["Strategy"]["Start Period"] == "2023-01-01"
        assert actual["Strategy"]["End Period"] == "2023-06-01"
        assert abs(actual["Strategy"]["Cumulative Return"] - 0.54) < 1e-3

    def test_plot_quantile_returns(self, medium_test_data):
        """Test plot_quantile_returns method doesn't raise exceptions and doesn't block"""
        factor_df, bars = medium_test_data
        ms = Moonshot(bars)

        # Append factor and run to generate quantile_returns data
        ms.append_factor(factor_df, "factor", quantiles=3, resample="first")
        strategy_returns = ms.run()

        # Test basic plot without saving
        try:
            ms.plot_quantile_returns(show_cumulative=True)
        except Exception as e:
            pytest.fail(f"plot_quantile_returns raised an exception: {e}")

        # Test plot with monthly returns
        try:
            ms.plot_quantile_returns(show_cumulative=False)
        except Exception as e:
            pytest.fail(
                f"plot_quantile_returns with monthly returns raised an exception: {e}"
            )

        # Test plot with custom parameters
        try:
            ms.plot_quantile_returns(
                figsize=(10, 6), title="自定义标题", font_family="Arial Unicode MS"
            )
        except Exception as e:
            pytest.fail(
                f"plot_quantile_returns with custom parameters raised an exception: {e}"
            )

        # Test plot with save path (using temp directory)
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "test_plot.png")
            try:
                ms.plot_quantile_returns(save_path=save_path)
                # Check if file was created
                assert os.path.exists(save_path)
            except Exception as e:
                pytest.fail(
                    f"plot_quantile_returns with save_path raised an exception: {e}"
                )

    def test_plot_cumulative_returns(self, medium_test_data):
        """Test plot_cumulative_returns method doesn't raise exceptions and doesn't block"""
        factor_df, bars = medium_test_data
        ms = Moonshot(bars)

        # Append factor and run to generate strategy and benchmark returns
        ms.append_factor(factor_df, "factor", quantiles=3, resample="first")
        strategy_returns, benchmark_returns = ms.run()

        # Test basic plot without saving
        try:
            ms.plot_cumulative_returns(strategy_returns, benchmark_returns)
        except Exception as e:
            pytest.fail(f"plot_strategy_vs_benchmark raised an exception: {e}")

        # Test plot with custom parameters
        try:
            ms.plot_cumulative_returns(
                strategy_returns,
                benchmark_returns,
                figsize=(10, 6),
                title="自定义标题",
                font_family="Arial Unicode MS",
            )
        except Exception as e:
            pytest.fail(
                f"plot_strategy_vs_benchmark with custom parameters raised an exception: {e}"
            )

        # Test plot with provided data
        try:
            ms.plot_cumulative_returns(
                strategy=strategy_returns, benchmark=benchmark_returns
            )
        except Exception as e:
            pytest.fail(
                f"plot_strategy_vs_benchmark with provided data raised an exception: {e}"
            )

        # Test plot with save path (using temp directory)
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = os.path.join(tmpdir, "test_strategy_benchmark_plot.png")
            try:
                ms.plot_cumulative_returns(
                    strategy_returns, benchmark_returns, save_path=save_path
                )
                # Check if file was created
                assert os.path.exists(save_path)
            except Exception as e:
                pytest.fail(
                    f"plot_strategy_vs_benchmark with save_path raised an exception: {e}"
                )
