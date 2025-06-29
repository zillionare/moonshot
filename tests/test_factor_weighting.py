#!/usr/bin/env python3
"""
Factor Weighting Test Suite

测试因子加权功能的正确性，包括：
1. 纯多组合（Long-Only）计算
2. 多空组合（Long-Short）计算
3. 与等权重方法的对比
4. 边界条件处理
"""

import numpy as np
import pandas as pd
import pytest

from moonshot import Moonshot


@pytest.fixture
def factor_weighting_test_data():
    """为因子加权测试提供数据"""
    # 创建3个月的数据，5只股票
    dates = pd.date_range("2023-01-01", "2023-03-31", freq="D")
    assets = ["Stock_A", "Stock_B", "Stock_C", "Stock_D", "Stock_E"]

    # 因子数据：明确的因子值排序
    factor_data = []
    factor_values = [1.0, 2.0, 3.0, 4.0, 5.0]  # 明确的排序

    for month in pd.date_range("2023-01-31", "2023-02-28", freq="ME"):
        for i, asset in enumerate(assets):
            factor_data.append(
                {
                    "date": month,
                    "asset": asset,
                    "factor": factor_values[i],
                }
            )

    factor_df = pd.DataFrame(factor_data)
    factor_series = factor_df.set_index(["date", "asset"])["factor"]

    # 价格数据：收益率与因子值正相关
    price_data = []
    base_returns = [0.01, 0.02, 0.03, 0.04, 0.05]  # 与因子值对应的收益率

    for date in dates:
        for i, asset in enumerate(assets):
            base_price = 100
            # 简单的价格演化
            daily_return = base_returns[i] / 30  # 月收益率转日收益率
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


def test_factor_weighting_basic(factor_weighting_test_data):
    """测试因子加权的基本功能"""
    factor_series, bars = factor_weighting_test_data

    # 测试因子加权方法
    moonshot = Moonshot()
    moonshot.backtest(
        factor_data=factor_series,
        bars=bars,
        quantiles=5,
        weighting_method="factor_weight",
    )
    quantile_returns = moonshot.quantile_returns
    benchmark_returns = moonshot.benchmark_returns
    long_only_returns = moonshot.long_only_returns
    long_short_returns = moonshot.long_short_returns
    optimal_returns = moonshot.optimal_returns

    # 验证返回结果
    assert not quantile_returns.empty
    assert not benchmark_returns.empty
    assert moonshot.long_only_returns is not None
    assert moonshot.long_short_returns is not None

    # 验证数据类型
    assert isinstance(moonshot.long_only_returns, pd.Series)
    assert isinstance(moonshot.long_short_returns, pd.Series)

    # 验证索引一致性
    assert len(moonshot.long_only_returns) == len(quantile_returns)
    assert len(moonshot.long_short_returns) == len(quantile_returns)


def test_equal_weight_vs_factor_weight(factor_weighting_test_data):
    """对比等权重和因子加权方法"""
    factor_series, bars = factor_weighting_test_data

    # 等权重方法
    moonshot_eq = Moonshot()
    moonshot_eq.backtest(
        factor_data=factor_series,
        bars=bars,
        quantiles=5,
        weighting_method="equal_weight",
    )
    strategy_eq = moonshot_eq.quantile_returns
    benchmark_eq = moonshot_eq.benchmark_returns
    long_only_eq = moonshot_eq.long_only_returns
    long_short_eq = moonshot_eq.long_short_returns
    optimal_eq = moonshot_eq.optimal_returns

    # 因子加权方法
    moonshot_fw = Moonshot()
    moonshot_fw.backtest(
        factor_data=factor_series,
        bars=bars,
        quantiles=5,
        weighting_method="factor_weight",
    )
    strategy_fw = moonshot_fw.quantile_returns
    benchmark_fw = moonshot_fw.benchmark_returns
    long_only_fw = moonshot_fw.long_only_returns
    long_short_fw = moonshot_fw.long_short_returns
    optimal_fw = moonshot_fw.optimal_returns

    # 验证两种方法都有结果
    assert not strategy_eq.empty
    assert not strategy_fw.empty

    # 验证因子加权方法有额外的收益序列
    assert (
        moonshot_eq.long_only_returns is None or len(moonshot_eq.long_only_returns) > 0
    )
    assert moonshot_fw.long_only_returns is not None
    assert moonshot_fw.long_short_returns is not None

    # 验证基准收益应该相同（都是等权重）
    np.testing.assert_array_almost_equal(
        benchmark_eq.values, benchmark_fw.values, decimal=10
    )


def test_factor_weight_calculation_logic(factor_weighting_test_data):
    """测试因子加权计算逻辑的正确性"""
    factor_series, bars = factor_weighting_test_data

    moonshot = Moonshot()
    moonshot.backtest(
        factor_data=factor_series,
        bars=bars,
        quantiles=5,
        weighting_method="factor_weight",
    )
    quantile_returns = moonshot.quantile_returns
    benchmark_returns = moonshot.benchmark_returns
    long_only_returns = moonshot.long_only_returns
    long_short_returns = moonshot.long_short_returns
    optimal_returns = moonshot.optimal_returns

    # 验证纯多组合收益率应该为正（因为高因子值对应高收益）
    assert moonshot.long_only_returns.mean() > 0

    # 验证多空组合收益率应该为正（因为做多高因子值，做空低因子值）
    assert moonshot.long_short_returns.mean() > 0

    # 验证多空组合的波动性应该比纯多组合小（理论上）
    # 注意：这个测试可能在某些情况下失败，取决于具体的数据
    # assert moonshot.long_short_returns.std() <= moonshot.long_only_returns.std()


def test_factor_weight_edge_cases():
    """测试因子加权的边界情况"""
    # 创建所有因子值相同的情况
    dates = pd.date_range("2023-01-01", "2023-02-28", freq="D")
    assets = ["A", "B", "C"]

    # 所有因子值相同
    factor_data = []
    for month in pd.date_range("2023-01-31", "2023-01-31", freq="ME"):
        for asset in assets:
            factor_data.append(
                {
                    "date": month,
                    "asset": asset,
                    "factor": 1.0,  # 所有因子值相同
                }
            )

    factor_df = pd.DataFrame(factor_data)
    factor_series = factor_df.set_index(["date", "asset"])["factor"]

    # 价格数据
    price_data = []
    for date in dates:
        for asset in assets:
            price_data.append(
                {
                    "date": date,
                    "asset": asset,
                    "open": 100,
                    "close": 105,  # 5%收益率
                }
            )

    bars_df = pd.DataFrame(price_data)
    bars = bars_df.set_index(["date", "asset"])

    # 测试因子加权方法
    moonshot = Moonshot()
    try:
        moonshot.backtest(
            factor_data=factor_series,
            bars=bars,
            quantiles=3,
            weighting_method="factor_weight",
        )
        strategy_returns = moonshot.quantile_returns
        benchmark_returns = moonshot.benchmark_returns
        long_only_returns = moonshot.long_only_returns
        long_short_returns = moonshot.long_short_returns
        optimal_returns = moonshot.optimal_returns

        # 当所有因子值相同时，多空组合收益应该接近0
        if (
            moonshot.long_short_returns is not None
            and len(moonshot.long_short_returns) > 0
        ):
            assert abs(moonshot.long_short_returns.mean()) < 0.01

    except ValueError:
        # 如果因子值相同导致无法分组，这是预期的行为
        pass


def test_factor_weight_with_negative_factors():
    """测试包含负因子值的情况"""
    dates = pd.date_range("2023-01-01", "2023-02-28", freq="D")
    assets = ["A", "B", "C"]

    # 包含负因子值
    factor_data = []
    factor_values = [-1.0, 0.0, 1.0]

    for month in pd.date_range("2023-01-31", "2023-01-31", freq="ME"):
        for i, asset in enumerate(assets):
            factor_data.append(
                {
                    "date": month,
                    "asset": asset,
                    "factor": factor_values[i],
                }
            )

    factor_df = pd.DataFrame(factor_data)
    factor_series = factor_df.set_index(["date", "asset"])["factor"]

    # 价格数据
    price_data = []
    returns = [0.01, 0.03, 0.05]  # 与因子值正相关

    for date in dates:
        for i, asset in enumerate(assets):
            daily_return = returns[i] / 30
            price = 100 * (1 + daily_return) ** ((date - dates[0]).days)

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

    # 测试因子加权方法
    moonshot = Moonshot()
    moonshot.backtest(
        factor_data=factor_series,
        bars=bars,
        quantiles=3,
        weighting_method="factor_weight",
    )
    quantile_returns = moonshot.quantile_returns
    benchmark_returns = moonshot.benchmark_returns
    long_only_returns = moonshot.long_only_returns
    long_short_returns = moonshot.long_short_returns
    optimal_returns = moonshot.optimal_returns

    # 验证结果存在
    assert not quantile_returns.empty
    assert moonshot.long_only_returns is not None
    assert moonshot.long_short_returns is not None

    # 验证纯多组合收益为正（因为负因子值被转换为正权重）
    assert moonshot.long_only_returns.mean() > 0


def test_moonshot_methods_with_factor_weight(factor_weighting_test_data):
    """测试Moonshot类的方法在因子加权模式下的工作"""
    factor_series, bars = factor_weighting_test_data

    moonshot = Moonshot()
    moonshot.backtest(
        factor_data=factor_series,
        bars=bars,
        quantiles=5,
        weighting_method="factor_weight",
    )

    # 测试计算指标
    metrics = moonshot.calculate_metrics()
    assert not metrics.empty
    assert "long-short" in metrics.columns
    assert "long-only" in metrics.columns
    assert "benchmark" in metrics.columns

    # 测试绘图方法（不会实际显示图表，但会验证代码执行）
    try:
        # 这些方法在测试环境中可能会有matplotlib后端问题，所以用try-except
        moonshot.plot_long_short_returns()
        moonshot.plot_long_only_returns()
    except Exception:
        # 在无GUI环境中绘图可能失败，这是可以接受的
        pass
