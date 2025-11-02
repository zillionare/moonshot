from typing import Optional

import pandas as pd
from loguru import logger
from moonshot.metrics import Metrics
from moonshot.helper import (resample_to_month)




class Moonshot:
    """量化回测框架类

    实现股票筛选、回测和报告功能
    """

    def __init__(self, daily_bars:pd.DataFrame):
        """初始化Moonshot实例

        Args:
            daily_bars: 列字段包含date, asset以及open, close的已复权数据，其中date必须为datetime.date类型
        """
        self.data: pd.DataFrame = resample_to_month(daily_bars, open='first', close='last')
        self.data['flag'] = 1

        self.strategy_returns: Optional[pd.Series] = None
        self.benchmark_returns: Optional[pd.Series] = None
        self.metrics: Metrics|None = None


    def append_factor(self, data: pd.DataFrame, factor_col: str, resample_method: str|None=None) -> None:
        """将因子数据添加到回测数据(即self.data)中。

        如果resample_method参数不为None, 则需要重采样为月频，并且使用resample_method指定的方法。否则，认为因子已经是月频的，且以'month', 'asset'为索引 ，将直接添加到回测数据中。

        使用本方法，一次只能添加一个因子。

        Args:
            data: 因子数据，需包含'date'和'asset'列
            factor_col: 因子列名
            resample_method: 如果需要对因子重采样，此列为重采样方法。
        """
        # 检查必需的列是否存在
        if factor_col not in data.columns:
            raise ValueError(f"因子数据中不存在列: {factor_col}")

        if resample_method is not None:
            assert ('date' in data.columns and 'asset' in data.columns), "缺少'date'/'asset'列"
            factor_data = resample_to_month(data, **{factor_col: resample_method})
        else:
            factor_data = data[[factor_col]]

        self.data = self.data.join(factor_data, how='left')

    def screen(self, screen_method, **kwargs) -> 'Moonshot':
        """应用股票筛选器

        Args:
            screen_method: 筛选方法（可调用对象）
            **kwargs: 筛选器参数

        Returns:
            Moonshot: 返回自身以支持链式调用
        """
        if self.data is None or self.data.empty:
            raise ValueError("警告：数据为空，无法应用筛选器")

        if callable(screen_method):
            flags = screen_method(**kwargs)
            logger.info(f"{screen_method.__name__} 筛选结果：\n{self.data[self.data['flag'] == 1]}")

            # 当月选股，下月开仓
            flags = flags.groupby(level='asset').shift(1).fillna(0).astype(int)

            # 与现有flag进行逻辑与运算
            self.data['flag'] = self.data['flag'] & flags
        else:
            raise ValueError("screen_method 必须是可调用对象")

        return self

    def calculate_returns(self)->'Moonshot':
        """计算策略收益率和基准收益率（向量化实现）

        使用向量化操作计算：
        1. 策略收益：每月flag=1的股票的等权平均收益
        2. 基准收益：每月所有股票的等权平均收益
        """
        if self.data is None or self.data.empty:
            raise ValueError("警告：数据为空，无法计算收益率")

        # 计算所有股票的月收益率 (close - open) / open
        self.data['monthly_return'] = (self.data['close'] - self.data['open']) / self.data['open']

        # 按月分组计算策略收益（flag=1的股票等权平均）
        def calculate_strategy_return(group):
            long_positions = group[group.get('flag', 0) == 1]
            short_positions = group[group.get('flag', 0) == -1]

            long_return = 0.0
            short_return = 0.0

            if len(long_positions) > 0:
                long_return = long_positions['monthly_return'].mean()

            if len(short_positions) > 0:
                short_return = -short_positions['monthly_return'].mean()

            return long_return + short_return

        # 向量化计算策略收益
        strategy_returns = self.data.groupby(level='month').apply(calculate_strategy_return)
        strategy_returns.name = 'strategy_returns'

        # 向量化计算基准收益（所有股票等权平均）
        benchmark_returns = self.data.groupby(level='month')['monthly_return'].mean()
        benchmark_returns.name = 'benchmark_returns'

        # 将PeriodIndex转换为DatetimeIndex以兼容QuantStats
        if isinstance(strategy_returns.index, pd.PeriodIndex):
            strategy_returns.index = strategy_returns.index.to_timestamp()
        if isinstance(benchmark_returns.index, pd.PeriodIndex):
            benchmark_returns.index = benchmark_returns.index.to_timestamp()

        # 存储结果
        self.strategy_returns = strategy_returns
        self.benchmark_returns = benchmark_returns

        self.metrics = StrategyAnalyzer(
            strategy_returns=self.strategy_returns,
            benchmark_returns=self.benchmark_returns
        )

        return self


    def report(self, output: str = "strategy_report.html", title: str = "策略绩效报告"):
        """生成策略报告

        Args:
            output: 报告输出路径，默认为"strategy_report.html"
            title: 报告标题，默认为"策略绩效报告"
        """
        if self.metrics is None:
            raise ValueError("请先调用calculate_returns方法计算收益率")

        self.metrics.generate_html_report(output=output, title=title)

        return self
