from typing import List, Literal, Optional

import pandas as pd
import quantstats.reports
import quantstats.stats
from loguru import logger

from moonshot.helper import resample_to_month


class Moonshot:
    """量化回测框架类

    实现股票筛选、回测和报告功能
    """

    def __init__(self, daily_bars: pd.DataFrame):
        """初始化Moonshot实例

        Args:
            daily_bars: 列字段包含date, asset以及open, close的已复权数据，其中date必须为datetime.date类型
        """
        # 初始化数据和价格
        self.data: pd.DataFrame = resample_to_month(
            daily_bars, open="first", close="last"
        )

        # 初始化策略和分析器
        self.strategy_returns: Optional[pd.Series] = None
        self.benchmark_returns: Optional[pd.Series] = None
        self.quantile_returns: Optional[pd.DataFrame] = None

        # 初始化因子相关属性
        self.factor_names: List[str] = []  # 这些列会被当成因子
        self.factor_types: list[str] = []  # 记录每个因子的类型（'discrete'或'continuous'）
        self.factor_quantiles = []  # 记录每个连续因子的分层数

    def check_factor_type(self, factor: pd.Series) -> Literal["discrete", "continuous"]:
        """检查因子类型"""
        values = set(factor.unique())
        if set(values).issubset({-1, 0, 1}):
            return "discrete"

        return "continuous"

    def is_month_indexed(self, df: pd.DataFrame | pd.Series) -> bool:
        """检查DataFrame是否按月份索引"""
        return isinstance(df.index, pd.MultiIndex) and df.index.names == [
            "month",
            "asset",
        ]

    def append_factor(
        self,
        data: pd.DataFrame | pd.Series,
        factor_col: str,
        quantiles: Optional[int] = None,
        resample: str | None = None,
    ) -> None:
        """将因子数据添加到回测数据(即self.data)中。

        Args:
            data: 因子数据，需包含'date'和'asset'列
            factor_col: 因子列名
            quantiles: 对于连续因子，指定分层数量
            resample: 如果 data 未按月份索引，需要指定重采样方法，比如 first, last, max 等
        """
        if isinstance(data, pd.Series):
            data = data.to_frame(name=factor_col)

        if factor_col not in data.columns:
            raise ValueError(f"因子数据中不存在列: {factor_col}")

        if factor_col in self.data.columns:
            raise ValueError(f"重复加入因子：{factor_col}")

        self.factor_names.append(factor_col)

        factor_type = self.check_factor_type(data[factor_col])
        self.factor_types.append(factor_type)

        if factor_type == "discrete" and quantiles is not None:
            logger.warning("因子数据为离散型，quantiles参数将被忽略")

        self.factor_quantiles.append(quantiles)

        # 处理数据重采样
        factor_data = data.copy()
        if not self.is_month_indexed(factor_data):
            if resample is None:
                raise ValueError("数据未按月份索引，请使用resample_method参数指定重采样方法")

            if not set(["date", "asset"]).issubset(factor_data.columns):
                raise ValueError("重采样需要数据包含'date, asset'列")

            if not pd.api.types.is_datetime64_any_dtype(factor_data["date"]):
                factor_data["date"] = pd.to_datetime(factor_data["date"])

            factor_data = resample_to_month(factor_data, **{factor_col: resample})

        self.data[factor_col] = factor_data[factor_col]

    def run(self, long_only: bool = True) -> tuple[pd.Series, pd.Series]:
        """运行回测

        Args:
            long_only (bool, optional): 是否只允许做多。默认为 True。

        Returns:
            tuple[pd.Series, pd.Series]: 策略收益和基准收益，以月为单位。
        """
        # 验证是否有因子数据
        if len(self.factor_names) == 0:
            raise ValueError("请先使用 append_factor 方法添加因子数据")

        # 计算基准收益
        self.benchmark_returns = self._calculate_benchmark_returns()

        if len(self.factor_names) == 1:
            self.strategy_returns = self._calculate_single_factor_returns(long_only)
        else:
            self.strategy_returns = self._calculate_multiple_factors_returns(long_only)

        return self.strategy_returns, self.benchmark_returns

    def _calculate_benchmark_returns(self) -> pd.Series:
        """计算基准收益（买入并持有策略）

        第一个月收益是 close/open-1，此后每月为 close/prev(close)-1。因此返回的收益序列长度与 month 索引等长，很好地反映了作为基准，买入并持有的收益。

        Returns:
            pd.Series: 基准收益序列，以月为单位
        """
        # 计算第一个月的收益率
        first_month_returns = (
            (self.data["close"] / self.data["open"] - 1)
            .groupby(level="asset")
            .first()
            .mean()
        )

        # 后续使用 close
        prices = self.data["close"].unstack(level="asset")
        subsequent_returns = prices.pct_change().iloc[1:].mean(axis=1)

        # 合并收益率
        benchmark_returns = pd.concat(
            [
                pd.Series([first_month_returns], index=[prices.index[0]]),
                subsequent_returns,
            ]
        )

        return benchmark_returns

    def _discretize_continuous_factors(self) -> None:
        """对连续因子进行离散化处理

        将连续因子按月分层，将top层的因子设为1，bottom层因子设为-1，中间的因子设为0
        """
        # pylint: disable='consider-using-enumerate'
        for i in range(len(self.factor_names)):
            if self.factor_types[i] == "continuous":
                quantiles = self.factor_quantiles[i] or 5

                # 按月对因子进行分层
                factor_name = self.factor_names[i]  # 使用存储的原始因子列名

                # 使用qcut进行分层
                discretized = (
                    self.data.groupby(level="month")[factor_name]
                    .apply(
                        lambda x: pd.qcut(
                            x, q=quantiles, labels=False, duplicates="drop"
                        )
                    )
                    .droplevel(0)
                )

                # 将分层结果转换为-1, 0, 1
                # 最低层为-1，最高层为1，中间层为0
                max_quantile = discretized.max()
                min_quantile = discretized.min()

                # 创建新的离散化因子
                discretized_factor = discretized.copy()
                discretized_factor[discretized == max_quantile] = 1
                discretized_factor[discretized == min_quantile] = -1
                discretized_factor[
                    (discretized > min_quantile) & (discretized < max_quantile)
                ] = 0

                # 更新因子数据
                self.data[factor_name] = discretized_factor

    def _merge_factors_to_flag(self) -> pd.Series:
        """合并多个因子得到flag列

        如果所有因子列为1，则flag为1；所有因子列为-1，则flag为-1；其它为0

        Returns:
            pd.Series: flag列
        """
        factors = self.data[self.factor_names]
        all_1 = factors.apply(lambda x: (x == 1).all(), axis=1)
        all_minus_1 = factors.apply(lambda x: (x == -1).all(), axis=1)
        flag = pd.Series(0, index=factors.index)
        flag[all_1] = 1
        flag[all_minus_1] = -1
        return flag

    def _calculate_multiple_factors_returns(self, long_only: bool) -> pd.Series:
        """计算多因子策略收益

        Args:
            long_only: 是否只做多

        Returns:
            pd.Series: 策略收益序列
        """
        # 1. 对连续因子进行离散化
        self._discretize_continuous_factors()

        # 2. 合并因子得到flag
        flag = self._merge_factors_to_flag()

        # 3. 计算策略收益
        return self._calculate_flag_returns(flag, long_only)

    def _calculate_single_factor_returns(self, long_only: bool) -> pd.Series:
        """计算单因子策略收益

        Args:
            long_only: 是否只做多

        Returns:
            pd.Series: 策略收益序列
        """
        factor_name = self.factor_names[0]  # 使用存储的原始因子列名
        factor = self.data[factor_name]  # 从self.data获取因子数据
        factor_type = self.factor_types[0]

        if factor_type == "discrete":
            flag = factor
            return self._calculate_flag_returns(flag, long_only)
        else:
            return self._calculate_quantile_returns(long_only)

    def _calculate_flag_returns(self, flag: pd.Series, long_only: bool) -> pd.Series:
        """基于flag计算策略收益

        Args:
            flag: 因子flag序列
            long_only: 是否只做多

        Returns:
            pd.Series: 策略收益序列
        """
        assert self.is_month_indexed(flag), "flag序列必须是月度索引"

        data_with_flag = self.data.copy()
        # 当月的收益，归因到上月的因子
        data_with_flag["flag"] = flag.groupby(level="asset").shift(1)

        # 计算月度收益 (MOM)
        data_with_flag["mom_returns"] = data_with_flag.groupby(level="asset")[
            "close"
        ].pct_change()

        # 计算COO收益 (Close/Open-1)
        data_with_flag["coo_returns"] = (
            data_with_flag["close"] / data_with_flag["open"] - 1
        )

        # 计算is_new标记
        prev = data_with_flag.groupby("asset")["flag"].shift(1)
        prev = prev.where(prev.abs() == 1, 0)

        # 判断新开仓
        data_with_flag["is_new"] = (
            (data_with_flag["flag"].abs() == 1) & (data_with_flag["flag"] != prev)
        ).astype(bool)

        # 获取所有月份并排序
        months = sorted(data_with_flag.index.get_level_values("month").unique())

        # 计算每月收益
        monthly_returns = []

        divider = 1 if long_only else 2
        for month in months:
            month_data = data_with_flag.loc[month]

            # 计算多头收益
            long_returns = self._calculate_position_returns(month_data, position_type=1)

            # 计算空头收益
            if long_only:
                short_returns = 0
            else:
                short_returns = self._calculate_position_returns(
                    month_data, position_type=-1
                )

            # 组合收益
            monthly_returns.append((long_returns + short_returns) / divider)

        # 创建收益率序列
        strategy_returns = pd.Series(monthly_returns, index=months)

        return strategy_returns

    def _calculate_position_returns(
        self, month_data: pd.DataFrame, position_type: int
    ) -> float:
        """计算特定持仓类型的收益

        Args:
            month_data: 当月数据，包含flag, mom_returns, coo_returns, is_new列
            position_type: 持仓类型，1表示多头，-1表示空头

        Returns:
            float: 该持仓类型的平均收益
        """
        assert position_type in [1, -1], "position_type参数只能为1(多）或-1（空）"

        # 筛选符合条件的资产
        position_assets = month_data[month_data["flag"] == position_type]

        if position_assets.empty:
            return 0

        # 分离新买入和继续持有的资产
        new_assets = position_assets[position_assets["is_new"]]
        old_assets = position_assets[~position_assets["is_new"]]

        # 计算收益
        total_return = 0
        total_assets = len(position_assets)

        # 新买入资产的COO收益
        if not new_assets.empty:
            total_return += new_assets["coo_returns"].sum()

        # 继续持有资产的MOM收益
        if not old_assets.empty:
            total_return += old_assets["mom_returns"].sum()

        average_return = total_return / total_assets if total_assets > 0 else 0

        return average_return * position_type

    def _calculate_quantile_returns(self, long_only: bool) -> pd.Series:
        """计算连续因子的分层收益

        Args:
            long_only: 是否只做多

        Returns:
            pd.Series: 策略收益序列
        """
        factor_name = self.factor_names[0]
        factor = self.data[factor_name]
        quantiles = self.factor_quantiles[0] or 5

        # 将因子添加到数据中
        data_with_factor = self.data.copy()
        data_with_factor["factor"] = factor.groupby(level="asset").shift(1)

        # 计算月度收益 (MOM)
        data_with_factor["mom_returns"] = data_with_factor.groupby(level="asset")[
            "close"
        ].pct_change()

        # 按月分组并计算每层的收益
        def calculate_monthly_quantile_returns(group):
            group_quantiles = pd.qcut(
                group["factor"], q=quantiles, labels=False, duplicates="drop"
            )

            return group.groupby(group_quantiles)["mom_returns"].mean()

        # 应用函数并转换为DataFrame
        quantile_returns = (
            data_with_factor.groupby(level="month")
            .apply(calculate_monthly_quantile_returns)
            .unstack()
        )

        divider = 1 if long_only else 2

        if long_only:
            strategy_returns = quantile_returns.iloc[:, -1] / divider
        else:
            strategy_returns = (
                quantile_returns.iloc[:, -1] - quantile_returns.iloc[:, 0]
            ) / divider

        index = data_with_factor.index.levels[0]
        self.quantile_returns = pd.DataFrame(quantile_returns, index=index)
        self.quantile_returns.fillna(0, inplace=True)

        strategy_returns = pd.Series(strategy_returns, index=index)
        return strategy_returns.fillna(0)

    def get_core_metrics(
        self, strategy: pd.Series, benchmark: pd.Series | None = None, rf=0
    ) -> dict:
        """返回核心评估指标

        Args:
            strategy (pd.Series): 策略收益，由 run 方法返回
            benchmark (pd.Series): 基准收益，由 run 方法返回

        Returns:
            dict: 评估指标
        """
        strategy = strategy.to_timestamp()
        if benchmark is not None:
            benchmark = benchmark.to_timestamp()

        nv = (1 + strategy).cumprod() - 1
        metrics = {
            "cagr": quantstats.stats.cagr(strategy, rf=rf, periods=12),
            "sharpe": quantstats.stats.sharpe(strategy, rf=rf, periods=12),
            "mdd": quantstats.stats.max_drawdown(nv),
            "sortino": quantstats.stats.sortino(strategy, rf=rf, periods=12),
            "calmar": quantstats.stats.calmar(strategy),
            "win_rate": quantstats.stats.win_rate(strategy),
        }

        # 如果有基准数据，添加超额收益指标
        if benchmark is not None:
            metrics.update(
                {
                    "alpha": self.alpha(
                        strategy,
                        benchmark,
                        periods=12,
                    ),
                    "beta": self.beta(
                        strategy,
                        benchmark,
                        periods=12,
                    ),
                }
            )

        return metrics

    def alpha(
        self, strategy: pd.Series, benchmark: pd.Series | None = None, periods=12
    ) -> float:
        """quantstats 没有直接定义 alpha/beta，需要通过 greeks 来调用

        !!! warning:
            quantstats 中的 alpha 年化方法并不准确，但为了保持数据一致性，
            我们仍然使用该方法。因为在它的 report 中，也会出现 alpha
        """
        if isinstance(strategy.index, pd.PeriodIndex):
            strategy = strategy.to_timestamp()
        if benchmark is not None and isinstance(benchmark.index, pd.PeriodIndex):
            benchmark = benchmark.to_timestamp()

        greeks = quantstats.stats.greeks(
            strategy, benchmark, periods, prepare_returns=False
        )

        return greeks.to_dict().get("alpha", 0)  # type: ignore

    def beta(
        self, strategy: pd.Series, benchmark: pd.Series | None = None, periods=12
    ) -> float:
        """quantstats 没有直接定义 alpha/beta，需要通过 greeks 来调用"""
        if isinstance(strategy.index, pd.PeriodIndex):
            strategy = strategy.to_timestamp()
        if benchmark is not None and isinstance(benchmark.index, pd.PeriodIndex):
            benchmark = benchmark.to_timestamp()

        greeks = quantstats.stats.greeks(
            strategy, benchmark, periods, prepare_returns=False
        )

        return greeks.to_dict().get("beta", 0)  # type: ignore

    def report(
        self,
        strategy: pd.Series,
        benchmark: pd.Series | None = None,
        kind: Literal["html", "full", "basic", "metrics", "plots"] = "html",
        **kwargs,
    ):
        """生成回测报告

        Args:
            strategy: 策略收益序列
            benchmark: 基准收益序列
            kind: 报告类型,参见 quantstats.report
        """
        strategy = strategy.to_timestamp()
        if benchmark is not None:
            benchmark = benchmark.to_timestamp()

        func = getattr(quantstats.reports, kind)
        return func(strategy, benchmark, **kwargs)
