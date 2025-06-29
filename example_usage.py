#!/usr/bin/env python3
"""
示例：使用扩展后的moonshot.backtest方法

展示如何使用新增的返回值：
1. long_only策略收益
2. long_short策略收益
3. 最优分层收益
"""

import numpy as np
import pandas as pd

from moonshot import Moonshot

# 创建示例数据
np.random.seed(42)

# 生成日期范围
dates = pd.date_range("2020-01-01", "2023-12-31", freq="D")
assets = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]

# 生成因子数据
factor_data = []
for date in dates:
    for asset in assets:
        factor_data.append(
            {"date": date, "asset": asset, "factor": np.random.randn()}  # 随机因子值
        )

factor_df = pd.DataFrame(factor_data)
factor_series = factor_df.set_index(["date", "asset"])["factor"]

# 生成价格数据
bars_data = []
for date in dates:
    for asset in assets:
        # 模拟价格数据
        base_price = 100 + np.random.randn() * 10
        bars_data.append(
            {
                "date": date,
                "asset": asset,
                "open": base_price,
                "close": base_price * (1 + np.random.randn() * 0.02),  # 2%的日波动
            }
        )

bars_df = pd.DataFrame(bars_data)
bars = bars_df.set_index(["date", "asset"])

# 使用扩展后的backtest方法
moonshot = Moonshot()

# 执行回测，现在返回5个值
moonshot.backtest(factor_series, bars, quantiles=3)

# 从moonshot实例获取回测结果
strategy_returns = moonshot.quantile_returns
benchmark_returns = moonshot.benchmark_returns
long_only_returns = moonshot.long_only_returns
long_short_returns = moonshot.long_short_returns
optimal_returns = moonshot.optimal_returns

print("=== 回测结果概览 ===")
print(f"策略分组收益形状: {strategy_returns.shape}")
print(f"基准收益长度: {len(benchmark_returns)}")
print(f"Long Only收益长度: {len(long_only_returns)}")
print(f"Long Short收益长度: {len(long_short_returns)}")
print(f"最优分层收益长度: {len(optimal_returns)}")
print(f"最优分层名称: {optimal_returns.name}")

print("\n=== 各策略累计收益对比 ===")
print("策略分组最终累计收益:")
for col in strategy_returns.columns:
    final_return = (1 + strategy_returns[col]).prod() - 1
    print(f"  {col}: {final_return:.2%}")

final_benchmark = (1 + benchmark_returns).prod() - 1
final_long_only = (1 + long_only_returns).prod() - 1
final_long_short = (1 + long_short_returns).prod() - 1
final_optimal = (1 + optimal_returns).prod() - 1

print(f"基准最终累计收益: {final_benchmark:.2%}")
print(f"Long Only最终累计收益: {final_long_only:.2%}")
print(f"Long Short最终累计收益: {final_long_short:.2%}")
print(f"最优分层最终累计收益: {final_optimal:.2%}")

# 计算性能指标
print("\n=== 性能指标对比 ===")
metrics = moonshot.calculate_metrics()
print(metrics)

# 绘制对比图
print("\n=== 绘制图表 ===")
print("绘制各组收益图...")
moonshot.plot_cumulative_returns_by_quantiles()

print("绘制最优分层收益对比图...")
moonshot.plot_optimal_returns()

print("绘制Long Only收益图...")
moonshot.plot_long_only_returns()

print("\n=== 使用建议 ===")
print("1. long_only_returns: 适合评估纯多头策略表现")
print("2. long_short_returns: 适合评估多空对冲策略表现")
print("3. optimal_returns: 展示事后最优分层选择的表现")
print("4. 可以将这些收益与benchmark_returns进行对比分析")
