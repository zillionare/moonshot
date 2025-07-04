"""
Monthly Factor Backtesting Framework
Implements monthly rebalancing strategy: buy at the beginning of each month based on factor grouping from the end of the previous month, sell at the end of the month
"""

import os
from typing import Any, List, Optional, Tuple, Union, cast

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class Moonshot:
    """
    Monthly Factor Backtesting Framework Class

    提供完整的因子回测分析功能，包括：
    - 回测计算
    - 收益率绘图
    - 多空组合分析
    - 性能指标计算
    """

    def __init__(self) -> None:
        """初始化Moonshot回测对象"""
        self.quantile_returns: Optional[pd.DataFrame] = None
        self.benchmark_returns: Optional["pd.Series[float]"] = None
        self.long_only_returns: Optional["pd.Series[float]"] = None
        self.long_short_returns: Optional["pd.Series[float]"] = None
        self.optimal_returns: Optional["pd.Series[float]"] = None
        self.factor_data: Optional["pd.Series[float]"] = None
        self.bars_data: Optional[pd.DataFrame] = None
        self.quantiles: Optional[int] = None
        self.bins: Optional[Union[int, List[float]]] = None
        self.factor_lag: Optional[int] = None
        self.weighting_method: Optional[str] = None
        self.ic_series: Optional["pd.Series[float]"] = None

    def _calculate_portfolio_returns(
        self,
        month_data: pd.DataFrame,
        group_returns: "pd.Series[float]",
        factor_col: str,
        weighting_method: str,
    ) -> Tuple[float, float]:
        """
        计算long-only和多空组合收益率

        Args:
            month_data: 当月数据，包含因子值和收益率
            group_returns: 分组收益率
            factor_col: 因子列名
            weighting_method: 权重计算方法

        Returns:
            tuple: (long-only收益率, 多空组合收益率)
        """
        if weighting_method == "factor_weight":
            # long-only：因子值归一化作为权重
            factor_values = month_data[factor_col].values
            # 将因子值转换为正权重（处理负值）
            factor_weights = factor_values - np.min(np.asarray(factor_values)) + 1e-8
            factor_weights = factor_weights / factor_weights.sum()
            long_only_return = (month_data["return"] * factor_weights).sum()

            # 多空组合：因子值去均值化并归一化作为权重
            demeaned_factors = factor_values - np.mean(np.asarray(factor_values))
            # 归一化权重，使得多头权重和为1，空头权重和为-1
            positive_weights = np.maximum(demeaned_factors, 0)
            negative_weights = np.minimum(demeaned_factors, 0)

            if positive_weights.sum() > 0 and negative_weights.sum() < 0:
                positive_weights = positive_weights / positive_weights.sum()
                negative_weights = negative_weights / abs(negative_weights.sum())
                long_short_weights = positive_weights + negative_weights
                long_short_return = (month_data["return"] * long_short_weights).sum()
            else:
                long_short_return = 0.0
        else:
            # 等权重方法：使用最高分组作为纯多，最高减最低作为多空
            if len(group_returns) >= 2:
                long_only_return = group_returns.iloc[-1]  # 最高分组
                long_short_return = (
                    group_returns.iloc[-1] - group_returns.iloc[0]
                )  # 多空价差
            else:
                long_only_return = (
                    group_returns.iloc[0] if len(group_returns) > 0 else 0.0
                )
                long_short_return = 0.0

        return long_only_return, long_short_return

    def _build_trading_calendar(self, bars_df: pd.DataFrame) -> pd.DataFrame:
        """
        基于价格数据构建交易日历

        Args:
            bars_df: 价格数据DataFrame，包含date列

        Returns:
            交易日历DataFrame，包含year_month, month_start, month_end列
        """
        # 确保日期列为datetime类型
        bars_df = bars_df.copy()
        bars_df["date"] = pd.to_datetime(bars_df["date"])

        # 添加年月信息
        bars_df["year_month"] = bars_df["date"].dt.to_period("M")

        # 构建交易日历：每月的首末交易日
        trading_calendar = (
            bars_df.groupby("year_month")["date"].agg(["min", "max"]).reset_index()
        )
        trading_calendar.columns = ["year_month", "month_start", "month_end"]

        return trading_calendar

    def _process_single_month(
        self,
        current_trading_month: "pd.Series[Any]",
        factor_month: "pd.Series[Any]",
        factor_df: pd.DataFrame,
        bars_df: pd.DataFrame,
        factor_col: str,
        quantiles: Optional[int] = None,
        bins: Optional[Union[int, List[float]]] = None,
        weighting_method: str = "equal_weight",
    ) -> Optional[Tuple["pd.Series[float]", float, float, float, float]]:
        """
        处理单个月的回测逻辑

        Args:
            current_trading_month: 当前交易月信息
            factor_month: 因子计算月信息
            factor_df: 因子数据DataFrame
            bars_df: 价格数据DataFrame
            factor_col: 因子列名
            quantiles: 分位数分组数量
            bins: 自定义分组边界
            weighting_method: 权重计算方法，可选：
                - "equal_weight": 等权重（原有方法）
                - "factor_weight": 因子加权

        Returns:
            如果成功处理返回(组收益率Series, 基准收益率float, long-only收益率float, 多空组合收益率float, IC值float)，否则返回None
        """
        # 获取因子计算时点的数据（通常是月末）
        factor_date = factor_month["month_end"]
        factor_month_data = factor_df[(factor_df["date"] == factor_date)].copy()

        if len(factor_month_data) == 0:
            return None

        # 买入价格（当月月初开盘价）
        buy_date = current_trading_month["month_start"]
        buy_prices = bars_df[bars_df["date"] == buy_date][["asset", "open"]].copy()
        buy_prices.columns = ["asset", "price_buy"]

        # 卖出价格（当月月末收盘价）
        sell_date = current_trading_month["month_end"]
        sell_prices = bars_df[bars_df["date"] == sell_date][["asset", "close"]].copy()
        sell_prices.columns = ["asset", "price_sell"]

        if len(buy_prices) == 0 or len(sell_prices) == 0:
            return None

        # 合并数据
        month_data = factor_month_data.merge(buy_prices, on="asset", how="inner")
        month_data = month_data.merge(sell_prices, on="asset", how="inner")

        # 移除缺失数据的股票
        month_data = month_data.dropna(subset=[factor_col, "price_buy", "price_sell"])

        if len(month_data) == 0:
            return None

        # 因子分组
        try:
            if quantiles is not None:
                month_data["group"] = (
                    pd.qcut(
                        month_data[factor_col],
                        q=quantiles,
                        labels=False,
                        duplicates="raise",
                    )
                    + 1
                )
            else:
                assert bins is not None, "bins 不能为 None"
                month_data["group"] = (
                    pd.cut(
                        month_data[factor_col],
                        bins=bins,
                        labels=False,
                        include_lowest=True,
                    )
                    + 1
                )
        except ValueError:
            # 如果因子值相同导致无法分组，跳过该月
            return None

        # 计算个股收益率
        month_data["return"] = month_data["price_sell"] / month_data["price_buy"] - 1

        # 计算各组等权重收益率（保持原有逻辑）
        group_returns = month_data.groupby("group")["return"].mean()

        # 计算基准收益率（所有股票等权重）
        benchmark_return = month_data["return"].mean()

        # 计算long-only和多空组合收益率
        long_only_return, long_short_return = self._calculate_portfolio_returns(
            month_data, group_returns, factor_col, weighting_method
        )

        # 计算IC值（因子值与收益率的相关系数）
        ic_value = month_data[factor_col].corr(month_data["return"])
        if pd.isna(ic_value):
            ic_value = 0.0

        # 添加月份信息
        group_returns.name = current_trading_month["year_month"]

        return (
            group_returns,
            benchmark_return,
            long_only_return,
            long_short_return,
            ic_value,
        )

    def backtest(
        self,
        factor_data: "pd.Series[float]",
        bars: pd.DataFrame,
        quantiles: Optional[int] = 5,
        bins: Optional[Union[int, List[float]]] = None,
        factor_lag: int = 1,
        weighting_method: str = "equal_weight",
    ) -> None:
        """
        执行月度因子回测

        重新设计的回测框架，基于价格数据而非因子数据来确定交易日历，
        解决了原有设计中时间逻辑混乱的问题。

        策略逻辑：
        1. 基于上月末因子值对股票分组
        2. 在下月初买入，下月末卖出
        3. 计算各组合的月度收益率

        如果因子值或者价格数据在交易日期（月初或者月末）缺少数据，该资产将被从组合中排除。这有可能导致回测数据不足。因此，推荐做法是您确保传入的因子数据和价格数据，都包含所有交易日期的数据。

        Args:
            factor_data: 因子数据，双重索引 (date, asset) 的 Series
            bars: 价格数据，双重索引 (date, asset)，包含 open 和 close 列
            quantiles: 分组数量，与 bins 互斥
            bins: 自定义分组边界，与 quantiles 互斥
            factor_lag: 因子滞后期（月），默认为1，表示用上月末因子预测下月收益
            weighting_method: 权重计算方法，可选：
                - "equal_weight": 等权重分组回测（原有方法）
                - "factor_weight": 因子加权回测（Alphalens风格）
        """
        # 保存参数
        self.factor_data = factor_data
        self.bars_data = bars
        self.quantiles = quantiles
        self.bins = bins
        self.factor_lag = factor_lag
        self.weighting_method = weighting_method

        # 执行回测
        result = self._monthly_factor_backtest(
            factor_data, bars, quantiles, bins, factor_lag, weighting_method
        )

        self.quantile_returns = result[0]
        self.benchmark_returns = result[1]
        self.long_only_returns = result[2]
        self.long_short_returns = result[3]
        self.ic_series = result[4]

        # 计算最优分层收益：选择累计收益最好的分层
        if self.quantile_returns is not None and len(self.quantile_returns.columns) > 0:
            # 计算各分层的累计收益
            cumulative_returns = (1 + self.quantile_returns).cumprod()
            # 选择最终累计收益最高的分层
            best_group = cumulative_returns.iloc[-1].idxmax()
            self.optimal_returns = self.quantile_returns[best_group].copy()
            self.optimal_returns.name = f"Optimal_{best_group}"
        else:
            self.optimal_returns = pd.Series(dtype=float, name="Optimal")

    def plot_long_short_returns(self, title: str = "long-short cumulative") -> None:
        """
        绘制多空组合累计净值图（叠加benchmark）

        Args:
            title: 图表标题
        """
        if self.quantile_returns is None or self.benchmark_returns is None:
            raise ValueError("请先执行backtest()方法")

        # 根据权重方法选择多空组合收益
        if (
            self.weighting_method == "factor_weight"
            and self.long_short_returns is not None
        ):
            long_short_spread = self.long_short_returns
        else:
            # 使用传统方法计算多空价差
            long_short_spread = self.analyze_long_short_spread(self.quantile_returns)

        # 计算累计收益
        long_short_cumulative = (1 + long_short_spread).cumprod()
        benchmark_cumulative = (1 + self.benchmark_returns).cumprod()

        plt.figure(figsize=(12, 8))
        plt.plot(
            long_short_cumulative.index,
            long_short_cumulative.values.astype(float),
            label="long-short cumulative",
            linewidth=2,
            color="red",
        )
        if self.benchmark_returns is not None:
            benchmark_cumulative = (1 + self.benchmark_returns).cumprod()
            plt.plot(
                benchmark_cumulative.index,
                benchmark_cumulative.values.astype(float),
                label="benchmark",
                linewidth=2,
                color="blue",
            )

        plt.title(title, fontsize=14)
        plt.xlabel("date", fontsize=12)
        plt.ylabel("cumulative net values", fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_optimal_returns(self, title: str = "top quantile cumulative") -> None:
        """
        绘制最优分层收益与其他策略的对比图

        Args:
            title: 图表标题
        """
        if (
            self.optimal_returns is None
            or self.benchmark_returns is None
            or self.long_only_returns is None
            or self.long_short_returns is None
        ):
            raise ValueError("请先执行backtest()方法")

        # 计算累计收益
        optimal_cumulative = (1 + self.optimal_returns).cumprod()
        benchmark_cumulative = (1 + self.benchmark_returns).cumprod()
        long_only_cumulative = (1 + self.long_only_returns).cumprod()
        long_short_cumulative = (1 + self.long_short_returns).cumprod()

        plt.figure(figsize=(12, 8))
        plt.plot(
            optimal_cumulative.index,
            optimal_cumulative.values.astype(float),
            label="optimal quantile cumulative",
            linewidth=2,
            color="red",
        )
        plt.plot(
            long_only_cumulative.index,
            long_only_cumulative.values.astype(float),
            label="Long Only",
            linewidth=2,
            color="green",
        )
        plt.plot(
            long_short_cumulative.index,
            long_short_cumulative.values.astype(float),
            label="Long Short",
            linewidth=2,
            color="orange",
        )
        plt.plot(
            benchmark_cumulative.index,
            benchmark_cumulative.values.astype(float),
            label="benchmark",
            linewidth=2,
            color="blue",
        )

        plt.title(title, fontsize=14)
        plt.xlabel("date", fontsize=12)
        plt.ylabel("net value", fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_long_only_returns(self, title: str = "long-only cumulative") -> None:
        """
        绘制long-only累计净值图（叠加benchmark）

        Args:
            title: 图表标题
        """
        if self.quantile_returns is None or self.benchmark_returns is None:
            raise ValueError("请先执行backtest()方法")

        # 根据权重方法选择long-only收益
        if (
            self.weighting_method == "factor_weight"
            and self.long_only_returns is not None
        ):
            long_only_returns = self.long_only_returns
        else:
            # 使用最高分组作为long-only
            long_only_returns = self.quantile_returns.iloc[:, -1]

        # 计算累计收益
        long_only_cumulative = (1 + long_only_returns).cumprod()
        if self.benchmark_returns is not None:
            benchmark_cumulative = (1 + self.benchmark_returns).cumprod()
        else:
            benchmark_cumulative = None

        plt.figure(figsize=(12, 8))
        plt.plot(
            long_only_cumulative.index,
            long_only_cumulative.values.astype(float),
            label=f"long-only ({self.weighting_method})",
            linewidth=2,
            color="green",
        )
        if benchmark_cumulative is not None:
            plt.plot(
                benchmark_cumulative.index,
                benchmark_cumulative.values.astype(float),
                label="benchmark",
                linewidth=2,
                color="blue",
            )

        plt.title(title, fontsize=14)
        plt.xlabel("date", fontsize=12)
        plt.ylabel("cumulative values", fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def calculate_metrics(self) -> pd.DataFrame:
        """
        计算性能指标并与benchmark对照

        Returns:
            包含各种指标的DataFrame，多列（策略组合，基准）多行（指标）
        """
        if self.quantile_returns is None or self.benchmark_returns is None:
            raise ValueError("请先执行backtest()方法")

        # 根据权重方法选择收益序列
        if (
            self.weighting_method == "factor_weight"
            and self.long_short_returns is not None
        ):
            long_short_spread = self.long_short_returns
            long_only_returns = self.long_only_returns
        else:
            # 使用传统方法计算
            long_short_spread = self.analyze_long_short_spread(self.quantile_returns)
            long_only_returns = self.quantile_returns.iloc[:, -1]

        # 计算指标
        metrics_data = {}

        # 策略收益使用多空组合
        metrics_data["strategy"] = self._calculate_single_metrics(long_short_spread)

        # 多空组合指标（保持向后兼容）
        metrics_data["long-short"] = self._calculate_single_metrics(long_short_spread)

        # long-only指标
        if long_only_returns is not None:
            metrics_data["long-only"] = self._calculate_single_metrics(
                long_only_returns
            )

        # 最优分层指标
        if self.optimal_returns is not None:
            metrics_data["optimal"] = self._calculate_single_metrics(
                self.optimal_returns
            )

        # 基准指标
        if self.benchmark_returns is not None:
            metrics_data["benchmark"] = self._calculate_single_metrics(
                self.benchmark_returns
            )

        return pd.DataFrame(metrics_data)

    def _calculate_single_metrics(
        self, returns: "pd.Series[float]"
    ) -> "dict[str, float]":
        """
        计算单个序列的性能指标

        Args:
            returns: 月度收益率序列

        Returns:
            包含各种指标的字典
        """
        # Ann. Returns
        annual_return = returns.mean() * 12

        # Ann. Vol
        annual_volatility = returns.std() * np.sqrt(12)

        # Sharpe（假设无风险利率为0）
        sharpe_ratio = (
            annual_return / annual_volatility if annual_volatility != 0 else 0
        )

        # Max DrawDonw
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()

        # Win Rate
        win_rate = (returns > 0).mean()

        # Sortino比率（下行风险调整收益）
        downside_returns = returns[returns < 0]
        downside_volatility = (
            downside_returns.std() * np.sqrt(12) if len(downside_returns) > 0 else 0
        )
        sortino_ratio = (
            annual_return / downside_volatility if downside_volatility != 0 else 0
        )

        # Calmar比率（年化收益/Max DrawDonw）
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0

        # CAGR（复合年增长率）
        total_periods = len(returns)
        total_return = cumulative_returns.iloc[-1] - 1
        cagr = (1 + total_return) ** (12 / total_periods) - 1

        # Alpha和Beta（相对于基准）
        if hasattr(self, "benchmark_returns") and self.benchmark_returns is not None:
            # 确保时间对齐
            aligned_returns = returns.reindex(self.benchmark_returns.index).dropna()
            aligned_benchmark = self.benchmark_returns.reindex(aligned_returns.index)

            if len(aligned_returns) > 1 and len(aligned_benchmark) > 1:
                covariance = np.cov(aligned_returns, aligned_benchmark)[0, 1]
                benchmark_variance = np.var(aligned_benchmark)
                beta = covariance / benchmark_variance if benchmark_variance != 0 else 0
                alpha = annual_return - beta * aligned_benchmark.mean() * 12
            else:
                alpha, beta = 0, 0
        else:
            alpha, beta = 0, 0

        # IC计算（Information Coefficient）
        ic_value = 0
        if (
            hasattr(self, "ic_series")
            and self.ic_series is not None
            and len(self.ic_series) > 0
        ):
            ic_value = self.ic_series.mean()

        return {
            "Ann. Returns": annual_return,
            "Ann. Vol": annual_volatility,
            "Sharpe": sharpe_ratio,
            "Max DrawDonw": max_drawdown,
            "Win Rate": win_rate,
            "Sortino": sortino_ratio,
            "Calmar": calmar_ratio,
            "CAGR": cagr,
            "Alpha": alpha,
            "Beta": beta,
            "IC": ic_value,
        }

    def calculate_group_statistics(self, monthly_returns: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate statistical indicators for each group

        Args:
            monthly_returns: Monthly returns DataFrame

        Returns:
            Statistical indicators DataFrame
        """
        stats = pd.DataFrame(index=monthly_returns.columns)

        # Annualized returns
        stats["Annualized_Return"] = monthly_returns.mean() * 12

        # Annualized volatility
        stats["Annualized_Volatility"] = monthly_returns.std() * np.sqrt(12)

        # Sharpe ratio (assuming risk-free rate is 0)
        stats["Sharpe_Ratio"] = (
            stats["Annualized_Return"] / stats["Annualized_Volatility"]
        )

        # Maximum drawdown
        cumulative_returns = (1 + monthly_returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        stats["Max_Drawdown"] = drawdown.min()

        # Win rate
        stats["Win_Rate"] = (monthly_returns > 0).mean()

        # Monthly return statistics
        stats["Monthly_Mean_Return"] = monthly_returns.mean()
        stats["Monthly_Return_Std"] = monthly_returns.std()

        return stats

    def plot_cumulative_returns_by_quantiles(
        self, title: str = "Cumulative Returns by quantiles"
    ) -> None:
        """
        Plot cumulative return curves for each group

        Args:
            title: Chart title
        """
        if self.quantile_returns is None:
            raise ValueError(
                "No quantile returns data available. Please run backtest first."
            )

        cumulative_returns = (1 + self.quantile_returns).cumprod()

        plt.figure(figsize=(12, 8))
        for col in cumulative_returns.columns:
            plt.plot(
                cumulative_returns.index,
                cumulative_returns[col],
                label=col,
                linewidth=2,
            )

        plt.title(title, fontsize=14)
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Cumulative Returns", fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def analyze_long_short_spread(
        self, monthly_returns: pd.DataFrame
    ) -> "pd.Series[float]":
        """
        Analyze long-short spread (highest group - lowest group)

        Args:
            monthly_returns: Monthly returns DataFrame

        Returns:
            Monthly returns Series of long-short spread
        """
        if len(monthly_returns.columns) < 2:
            raise ValueError(
                "At least 2 groups are required to calculate long-short spread"
            )

        highest_group = monthly_returns.iloc[:, -1]  # Last column (highest group)
        lowest_group = monthly_returns.iloc[:, 0]  # First column (lowest group)

        spread_returns = highest_group - lowest_group
        spread_returns.name = "Long_Short_Spread"

        return spread_returns

    def _monthly_factor_backtest(
        self,
        factor_data: "pd.Series[float]",
        bars: pd.DataFrame,
        quantiles: Optional[int] = 5,
        bins: Optional[Union[int, List[float]]] = None,
        factor_lag: int = 1,
        weighting_method: str = "equal_weight",
    ) -> Tuple[
        pd.DataFrame, "pd.Series[float]", "pd.Series[float]", "pd.Series[float]"
    ]:
        """
        Monthly Factor Backtesting Framework

        重新设计的回测框架，基于价格数据而非因子数据来确定交易日历，
        解决了原有设计中时间逻辑混乱的问题。

        策略逻辑：
        1. 基于上月末因子值对股票分组
        2. 在下月初买入，下月末卖出
        3. 计算各组合的月度收益率

        如果因子值或者价格数据在交易日期（月初或者月末）缺少数据，该资产将被从组合中排除。这有可能导致回测数据不足。因此，推荐做法是您确保传入的因子数据和价格数据，都包含所有交易日期的数据。

        Args:
            factor_data: 因子数据，双重索引 (date, asset) 的 Series
            bars: 价格数据，双重索引 (date, asset)，包含 open 和 close 列
            quantiles: 分组数量，与 bins 互斥
            bins: 自定义分组边界，与 quantiles 互斥
            factor_lag: 因子滞后期（月），默认为1，表示用上月末因子预测下月收益
            weighting_method: 权重计算方法，可选：
                - "equal_weight": 等权重分组回测（原有方法）
                - "factor_weight": 因子加权回测（Alphalens风格）

        Returns:
            tuple: (策略分组月度收益DataFrame, 基准月度收益Series, long-only收益Series, 多空组合收益Series, IC序列Series)
                   策略收益以月份为索引，分组为列
                   基准收益为所有股票等权重收益
                   纯多和多空组合收益根据weighting_method计算
                   IC序列为每月因子值与收益率的相关系数
        """

        # 参数验证
        if quantiles is not None and bins is not None:
            raise ValueError("quantiles 和 bins 不能同时指定")
        if quantiles is None and bins is None:
            quantiles = 5
        if factor_lag < 1:
            raise ValueError("factor_lag 必须大于等于1")

        # 重置索引便于操作
        if isinstance(factor_data, pd.Series):
            factor_data = factor_data.to_frame(name="factor")
        factor_df = factor_data.reset_index()
        factor_col = "factor"
        bars_df = bars.reset_index()

        # 检查数据是否为空
        if len(factor_df) == 0 or len(bars_df) == 0:
            return (
                pd.DataFrame(),
                pd.Series(dtype=float),
                pd.Series(dtype=float),
                pd.Series(dtype=float),
                pd.Series(dtype=float),
            )

        # 转换日期列为 datetime 类型
        factor_df["date"] = pd.to_datetime(factor_df["date"])
        bars_df["date"] = pd.to_datetime(bars_df["date"])

        # 构建交易日历
        trading_calendar = self._build_trading_calendar(bars_df)

        # 为因子数据添加年月信息
        factor_df["year_month"] = factor_df["date"].dt.to_period("M")

        # 存储月度收益
        monthly_returns = []
        benchmark_returns = []
        long_only_returns = []
        long_short_returns = []
        ic_values = []

        # 遍历交易日历，执行回测
        for i in range(factor_lag, len(trading_calendar)):
            current_trading_month = trading_calendar.iloc[i]
            factor_month = trading_calendar.iloc[i - factor_lag]

            # 处理单个月的回测逻辑
            result = self._process_single_month(
                current_trading_month=current_trading_month,
                factor_month=factor_month,
                factor_df=factor_df,
                bars_df=bars_df,
                factor_col=factor_col,
                quantiles=quantiles,
                bins=bins,
                weighting_method=weighting_method,
            )

            if result is not None:
                (
                    group_returns,
                    benchmark_return,
                    long_only_return,
                    long_short_return,
                    ic_value,
                ) = result
                monthly_returns.append(group_returns)
                benchmark_returns.append(benchmark_return)
                long_only_returns.append(long_only_return)
                long_short_returns.append(long_short_return)
                ic_values.append(ic_value)

        # 合并所有月份的收益
        if not monthly_returns:
            return pd.DataFrame(), pd.Series(), pd.Series(), pd.Series(), pd.Series()

        # 策略收益
        quantile_returns = pd.concat(monthly_returns, axis=1).T

        quantile_returns.index = cast(
            pd.PeriodIndex, quantile_returns.index
        ).to_timestamp(how="end", freq="D")

        # 重命名列
        if quantiles is not None:
            quantile_returns.columns = [f"Q{i}" for i in quantile_returns.columns]
        else:
            quantile_returns.columns = [f"Bin{i}" for i in quantile_returns.columns]

        # 基准收益
        benchmark_series = pd.Series(
            benchmark_returns, index=quantile_returns.index, name="Benchmark"
        )

        # long-only收益
        long_only_series = pd.Series(
            long_only_returns, index=quantile_returns.index, name="Long_Only"
        )

        # 多空组合收益
        long_short_series = pd.Series(
            long_short_returns, index=quantile_returns.index, name="Long_Short"
        )

        # IC序列
        ic_series = pd.Series(ic_values, index=quantile_returns.index, name="IC")

        return (
            quantile_returns,
            benchmark_series,
            long_only_series,
            long_short_series,
            ic_series,
        )


if __name__ == "__main__":
    moonshot = Moonshot()
    import pickle

    factor = pickle.load(open("/tmp/factor_5000.pkl", "rb"))
    barss = pickle.load(open("/tmp/barss_5000.pkl", "rb"))

    moonshot.backtest(factor, barss)
