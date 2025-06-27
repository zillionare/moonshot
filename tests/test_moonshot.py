#!/usr/bin/env python
"""Tests for `moonshot` package."""
# pylint: disable=redefined-outer-name

import pytest

#!/usr/bin/env python3
"""
月度因子回测框架测试方案

测试目标：
1. 验证函数逻辑正确性
2. 验证数值计算准确性
3. 验证边界条件处理
4. 验证数据完整性
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from monthly_factor_backtest import monthly_factor_backtest


class TestMonthlyFactorBacktest:
    """月度因子回测测试类"""

    def setup_method(self):
        """设置测试数据"""
        # 创建简单的测试数据
        self.dates = pd.date_range("2023-01-01", "2023-03-31", freq="D")
        self.assets = ["A", "B", "C", "D"]

        # 构造因子数据（月末数据）
        factor_dates = ["2023-01-31", "2023-02-28"]
        factor_data = []
        for date in factor_dates:
            for i, asset in enumerate(self.assets):
                factor_data.append(
                    {
                        "date": pd.to_datetime(date),
                        "asset": asset,
                        "factor": i + 1,  # A=1, B=2, C=3, D=4
                    }
                )

        factor_df = pd.DataFrame(factor_data)
        self.factor_data = factor_df.set_index(["date", "asset"])["factor"]

        # 构造价格数据
        price_data = []
        for date in self.dates:
            for asset in self.assets:
                price_data.append(
                    {
                        "date": date,
                        "asset": asset,
                        "open": 100,  # 简化：所有开盘价都是100
                        "close": 110
                        if asset == "A"
                        else 105
                        if asset == "B"
                        else 100
                        if asset == "C"
                        else 95,  # 不同收益率
                    }
                )

        bars_df = pd.DataFrame(price_data)
        self.bars = bars_df.set_index(["date", "asset"])

    def test_basic_functionality(self):
        """测试基本功能"""
        strategy_returns, benchmark_returns = monthly_factor_backtest(
            self.factor_data, self.bars, quantiles=2
        )

        # 验证返回值类型
        assert isinstance(strategy_returns, pd.DataFrame)
        assert isinstance(benchmark_returns, pd.Series)

        # 验证列名
        assert list(strategy_returns.columns) == ["Q1", "Q2"]
        assert benchmark_returns.name == "Benchmark"

        # 验证数据长度
        assert len(strategy_returns) == 1  # 只有一个完整的月度周期
        assert len(benchmark_returns) == 1

    def test_manual_calculation_verification(self):
        """手动计算验证数值正确性"""
        strategy_returns, benchmark_returns = monthly_factor_backtest(
            self.factor_data, self.bars, quantiles=2
        )

        # 手动计算预期结果
        # 因子值：A=1, B=2, C=3, D=4
        # 分组：Q1=[A,B], Q2=[C,D]
        # 收益率：A=10%, B=5%, C=0%, D=-5%

        expected_q1_return = (0.10 + 0.05) / 2  # A和B的平均收益率
        expected_q2_return = (0.00 + (-0.05)) / 2  # C和D的平均收益率
        expected_benchmark = (0.10 + 0.05 + 0.00 + (-0.05)) / 4  # 所有股票平均

        # 验证计算结果
        np.testing.assert_almost_equal(
            strategy_returns.iloc[0]["Q1"], expected_q1_return, decimal=6
        )
        np.testing.assert_almost_equal(
            strategy_returns.iloc[0]["Q2"], expected_q2_return, decimal=6
        )
        np.testing.assert_almost_equal(
            benchmark_returns.iloc[0], expected_benchmark, decimal=6
        )

    def test_custom_bins(self):
        """测试自定义分组边界"""
        # 使用自定义边界：[0, 2.5, 5]
        strategy_returns, _ = monthly_factor_backtest(
            self.factor_data, self.bars, quantiles=None, bins=[0, 2.5, 5]
        )

        # 验证列名
        assert list(strategy_returns.columns) == ["Bin1", "Bin2"]

        # 验证分组逻辑：Bin1=[A,B], Bin2=[C,D]
        expected_bin1 = (0.10 + 0.05) / 2
        expected_bin2 = (0.00 + (-0.05)) / 2

        np.testing.assert_almost_equal(
            strategy_returns.iloc[0]["Bin1"], expected_bin1, decimal=6
        )
        np.testing.assert_almost_equal(
            strategy_returns.iloc[0]["Bin2"], expected_bin2, decimal=6
        )

    def test_edge_cases(self):
        """测试边界条件"""
        # 测试空数据
        empty_factor = pd.Series([], dtype=float, name="factor")
        empty_factor.index = pd.MultiIndex.from_tuples([], names=["date", "asset"])
        empty_bars = pd.DataFrame(columns=["open", "close"])
        empty_bars.index = pd.MultiIndex.from_tuples([], names=["date", "asset"])

        strategy_returns, benchmark_returns = monthly_factor_backtest(
            empty_factor, empty_bars
        )

        assert strategy_returns.empty
        assert benchmark_returns.empty

    def test_missing_data_handling(self):
        """测试缺失数据处理"""
        # 创建有缺失价格数据的测试用例
        incomplete_bars = self.bars.copy()
        # 移除某些股票的价格数据
        incomplete_bars = incomplete_bars.drop(("2023-02-01", "A"))

        strategy_returns, benchmark_returns = monthly_factor_backtest(
            self.factor_data, incomplete_bars, quantiles=2
        )

        # 应该能正常运行，只是参与计算的股票数量减少
        assert not strategy_returns.empty
        assert not benchmark_returns.empty

    def test_parameter_validation(self):
        """测试参数验证"""
        # 测试同时指定quantiles和bins
        with pytest.raises(ValueError, match="quantiles和bins不能同时指定"):
            monthly_factor_backtest(
                self.factor_data, self.bars, quantiles=5, bins=[1, 2, 3]
            )


def create_realistic_test_data():
    """创建更真实的测试数据"""
    # 生成6个月的日度数据
    dates = pd.date_range("2023-01-01", "2023-06-30", freq="D")
    assets = [f"Stock_{i:03d}" for i in range(100)]  # 100只股票

    # 生成随机因子数据（月末）
    np.random.seed(42)
    factor_data = []
    for month in pd.date_range("2023-01-31", "2023-05-31", freq="ME"):
        for asset in assets:
            factor_data.append(
                {
                    "date": month,
                    "asset": asset,
                    "factor": np.random.normal(0, 1),  # 标准正态分布因子
                }
            )

    factor_df = pd.DataFrame(factor_data)
    factor_series = factor_df.set_index(["date", "asset"])["factor"]

    # 生成价格数据（简化模型：收益率与因子正相关）
    price_data = []
    for date in dates:
        for asset in assets:
            # 基础价格
            base_price = 100
            # 添加随机波动
            daily_return = np.random.normal(0, 0.02)  # 2%日波动率

            price_data.append(
                {
                    "date": date,
                    "asset": asset,
                    "open": base_price * (1 + daily_return),
                    "close": base_price
                    * (1 + daily_return + np.random.normal(0, 0.01)),
                }
            )

    bars_df = pd.DataFrame(price_data)
    bars_series = bars_df.set_index(["date", "asset"])

    return factor_series, bars_series


def test_realistic_scenario():
    """真实场景测试"""
    factor_data, bars_data = create_realistic_test_data()

    # 运行回测
    strategy_returns, benchmark_returns = monthly_factor_backtest(
        factor_data, bars_data, quantiles=5
    )

    print("=== 真实场景测试结果 ===")
    print(f"策略收益率形状: {strategy_returns.shape}")
    print(f"基准收益率长度: {len(benchmark_returns)}")
    print("\n策略各分组月度收益率:")
    print(strategy_returns)
    print("\n基准月度收益率:")
    print(benchmark_returns)

    # 验证基本合理性
    assert not strategy_returns.empty
    assert not benchmark_returns.empty
    assert len(strategy_returns.columns) == 5
    assert all(
        col in ["Q1", "Q2", "Q3", "Q4", "Q5"] for col in strategy_returns.columns
    )

    # 验证收益率在合理范围内（-50%到50%）
    assert strategy_returns.abs().max().max() < 0.5
    assert benchmark_returns.abs().max() < 0.5

    print("\n✅ 真实场景测试通过")


def test_benchmark_calculation():
    """专门测试基准收益计算"""
    # 创建两个月的数据用于回测
    dates = pd.date_range("2023-01-01", "2023-03-31", freq="D")
    assets = ["A", "B"]

    # 因子数据：包含1月和2月的完整交易日数据
    factor_dates = [
        pd.Timestamp("2023-01-01"),
        pd.Timestamp("2023-01-31"),  # 1月首末
        pd.Timestamp("2023-02-01"),
        pd.Timestamp("2023-02-28"),  # 2月首末
        pd.Timestamp("2023-03-01"),
        pd.Timestamp("2023-03-31"),  # 3月首末
    ]

    factor_data_list = []
    for date in factor_dates:
        factor_data_list.extend([(date, "A", 1.0), (date, "B", 2.0)])

    factor_data = pd.Series(
        [item[2] for item in factor_data_list],
        index=pd.MultiIndex.from_tuples(
            [(item[0], item[1]) for item in factor_data_list], names=["date", "asset"]
        ),
        name="factor",
    )

    # 价格数据：2月份A股票收益20%，B股票收益-10%
    price_data = []
    for date in dates:
        if date.month == 2:  # 2月份有收益
            price_data.extend(
                [
                    {"date": date, "asset": "A", "open": 100, "close": 120},  # 20%收益
                    {"date": date, "asset": "B", "open": 100, "close": 90},  # -10%收益
                ]
            )
        else:  # 其他月份无收益
            price_data.extend(
                [
                    {"date": date, "asset": "A", "open": 100, "close": 100},
                    {"date": date, "asset": "B", "open": 100, "close": 100},
                ]
            )

    bars_df = pd.DataFrame(price_data).set_index(["date", "asset"])

    # 运行回测
    strategy_returns, benchmark_returns = monthly_factor_backtest(
        factor_data, bars_df, quantiles=2
    )

    # 验证基准收益 = (20% + (-10%)) / 2 = 5%
    expected_benchmark = (0.20 + (-0.10)) / 2
    assert len(benchmark_returns) > 0, "基准收益率为空"
    np.testing.assert_almost_equal(
        benchmark_returns.iloc[0], expected_benchmark, decimal=6
    )

    print(f"基准收益率验证通过: {benchmark_returns.iloc[0]:.4f} ≈ {expected_benchmark:.4f}")
