# Moonshot 更新日志

## v2.0.0 - 扩展回测返回值

### 新增功能

#### 1. 扩展的backtest返回值

原来的`backtest`方法只返回2个值：
```python
strategy_returns, benchmark_returns = moonshot.backtest(factor_series, bars, quantiles=2)
```

现在返回5个值，提供更全面的策略分析：
```python
strategy_returns, benchmark_returns, long_only_returns, long_short_returns, optimal_returns = moonshot.backtest(
    factor_series, bars, quantiles=2
)
```

#### 2. 新增返回值说明

- **strategy_returns**: 策略分组月度收益DataFrame（原有）
- **benchmark_returns**: 基准月度收益Series（原有）
- **long_only_returns**: Long Only策略收益Series（新增）
  - 因子分层最高组的收益
  - 适合评估纯多头策略表现
- **long_short_returns**: Long Short策略收益Series（新增）
  - 最高组减最低组的收益
  - 适合评估多空对冲策略表现
- **optimal_returns**: 最优分层收益Series（新增）
  - 各个分层中累计收益最好的组合
  - 展示事后最优分层选择的表现

#### 3. 新增绘图方法

- `plot_optimal_returns()`: 绘制最优分层收益与其他策略的对比图
  - 同时展示最优分层、Long Only、Long Short和基准的累计收益曲线
  - 便于直观比较各策略表现

#### 4. 增强的性能指标计算

`calculate_metrics()`方法现在包含最优分层的性能指标：
- 年化收益率
- 年化波动率
- 夏普比率
- 最大回撤
- 胜率等

### 使用示例

```python
from moonshot import Moonshot

# 创建回测对象
moonshot = Moonshot()

# 执行回测
strategy_returns, benchmark_returns, long_only_returns, long_short_returns, optimal_returns = moonshot.backtest(
    factor_series, bars, quantiles=3
)

# 查看最优分层信息
print(f"最优分层: {optimal_returns.name}")
print(f"最优分层累计收益: {(1 + optimal_returns).prod() - 1:.2%}")

# 计算包含最优分层的性能指标
metrics = moonshot.calculate_metrics()
print(metrics)

# 绘制对比图
moonshot.plot_optimal_returns()
```

### 向后兼容性

**重要提醒**: 此版本更新了`backtest`方法的返回值数量，从2个增加到5个。

如果您的现有代码使用了解包赋值：
```python
# 旧代码 - 现在会报错
strategy_returns, benchmark_returns = moonshot.backtest(factor_series, bars)
```

请更新为：
```python
# 新代码 - 完整接收所有返回值
strategy_returns, benchmark_returns, long_only_returns, long_short_returns, optimal_returns = moonshot.backtest(
    factor_series, bars
)

# 或者只使用前两个返回值（如果不需要新功能）
results = moonshot.backtest(factor_series, bars)
strategy_returns, benchmark_returns = results[0], results[1]
```

### 技术改进

- 完善了类型注解，提高代码可维护性
- 优化了最优分层选择算法
- 增强了错误处理和边界情况处理
- 改进了文档和示例代码

### 文件变更

- `moonshot/moonshot.py`: 核心功能扩展
- `example_usage.py`: 新增使用示例
- `CHANGELOG.md`: 本更新日志
