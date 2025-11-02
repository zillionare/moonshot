import pandas as pd
import quantstats as qs
from typing import Optional


class Metrics:
    """策略分析器，通过代理模式集成QuantStats的分析功能

    !!! attention
        本分析器仅适用于 Moonshot，因为一些参数我们指定了默认值，且暂时未提供修改


    """
    def __init__(self, strategy_returns: pd.Series, benchmark_returns: Optional[pd.Series] = None, annual_periods: int = 12):
        """
        初始化分析器

        参数:
            strategy_returns: 策略收益率序列（索引为时间）
            benchmark_returns: 基准收益率序列（可选）
        """
        self.strategy_returns = self._validate_returns(strategy_returns, "策略收益率")
        self.benchmark_returns = self._validate_returns(benchmark_returns, "基准收益率") if benchmark_returns is not None else None

        self.annual_periods = annual_periods

        # 初始化QuantStats代理方法
        self._setup_proxy_methods()

    def _validate_returns(self, returns: pd.Series, name: str) -> pd.Series:
        """验证收益率序列的有效性"""
        if not isinstance(returns, pd.Series):
            raise TypeError(f"{name}必须是pandas Series类型")
        if returns.index.inferred_type not in ['datetime64', 'timedelta64', 'date', 'period']:
            raise ValueError(f"{name}的索引必须是时间类型，当前类型: {returns.index.inferred_type}")
        if returns.isna().any():
            raise ValueError(f"{name}中包含缺失值，请先处理")
        return returns

    def _setup_proxy_methods(self) -> None:
        """动态设置QuantStats的代理方法"""
        # 核心指标代理方法
        self.sharpe = lambda: qs.stats.sharpe(self.strategy_returns)
        self.max_drawdown = lambda: qs.stats.max_drawdown(self.strategy_returns)
        self.cagr = lambda: qs.stats.cagr(self.strategy_returns, periods = self.annual_periods)
        self.alpha = lambda: qs.stats.alpha(self.strategy_returns, self.benchmark_returns) if self.benchmark_returns is not None else None
        self.beta = lambda: qs.stats.volatility(self.strategy_returns) if self.benchmark_returns is not None else None

        # 报告生成代理方法
        def generate_html_report(output: str = "strategy_report.html", title: str = "策略绩效报告"):
            """生成HTML格式的完整报告"""
            qs.reports.html(
                self.strategy_returns,
                benchmark=self.benchmark_returns,
                output=output,
                title=title
            )

        def generate_tear_sheet():
            """生成简要的绩效分析摘要（适用于Jupyter Notebook）"""
            qs.reports.basic(
                self.strategy_returns,
                benchmark=self.benchmark_returns,
                title="策略绩效分析"
            )

        def plot_returns():
            """绘制策略累计收益率曲线"""
            qs.plots.returns(self.strategy_returns, benchmark=self.benchmark_returns)

        def plot_drawdown():
            """绘制最大回撤曲线"""
            qs.plots.drawdown(self.strategy_returns)

        # 绑定代理方法到当前实例
        self.generate_html_report = generate_html_report
        self.generate_tear_sheet = generate_tear_sheet
        self.plot_returns = plot_returns
        self.plot_drawdown = plot_drawdown

    def get_core_metrics(self) -> pd.DataFrame:
        """获取核心绩效指标汇总"""
        metrics = {
            "年化收益率": self.cagr(),
            "夏普比率": self.sharpe(),
            "最大回撤": self.max_drawdown(),
        }

        # 如果有基准数据，添加超额收益指标
        if self.benchmark_returns is not None:
            metrics.update({
                "阿尔法(α)": self.alpha(),
                "贝塔(β)": self.beta(),
                "信息比率": qs.stats.information_ratio(self.strategy_returns, self.benchmark_returns)
            })

        return pd.DataFrame(list(metrics.items()), columns=["指标", "值"])
