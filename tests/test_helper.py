import datetime
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_index_equal

from moonshot.helper import ensure_date, qfq_adjustment, resample_to_month


class TestEnsureDate:
    """测试ensure_date函数"""

    def test_ensure_date_with_string(self):
        """测试字符串日期转换"""
        date_str = "20231225"
        result = ensure_date(date_str)
        expected = datetime.date(2023, 12, 25)
        assert result == expected

    def test_ensure_date_with_datetime(self):
        """测试datetime对象转换"""
        dt = datetime.datetime(2023, 12, 25, 14, 30, 45)
        result = ensure_date(dt)
        expected = datetime.date(2023, 12, 25)
        assert result == expected

    def test_ensure_date_with_date(self):
        """测试date对象直接返回"""
        date_obj = datetime.date(2023, 12, 25)
        result = ensure_date(date_obj)
        assert result == date_obj

    def test_ensure_date_with_invalid_string_format(self):
        """测试无效字符串格式"""
        with pytest.raises(ValueError):
            ensure_date("2023-12-25")  # 错误的格式

    def test_ensure_date_with_unsupported_type(self):
        """测试不支持的类型"""
        with pytest.raises(ValueError) as exc_info:
            ensure_date(12345)
        assert "Unsupported date type" in str(exc_info.value)


class TestResampleToMonth:
    """测试resample_to_month函数"""

    @pytest.fixture
    def sample_data(self):
        """创建测试数据"""
        dates = pd.date_range("2023-01-01", "2023-01-31", freq="D")
        data = []
        for date in dates:
            data.append(
                {
                    "date": date,
                    "asset": "A",
                    "open": 100 + date.day,
                    "high": 110 + date.day,
                    "low": 90 + date.day,
                    "close": 105 + date.day,
                    "volume": 1000 * date.day,
                }
            )
        return pd.DataFrame(data)

    def test_resample_to_month_basic(self, sample_data):
        """测试基本重采样功能"""
        result = resample_to_month(
            sample_data, open="first", high="max", low="min", close="last", volume="sum"
        )

        assert len(result) == 1  # 一个月的数据
        assert result.index.names == ["month", "asset"]

        # 验证聚合结果
        row = result.iloc[0]
        assert row["open"] == 101  # 第一天
        assert row["high"] == 141  # 最后一天
        assert row["low"] == 91  # 第一天
        assert row["close"] == 136  # 最后一天
        assert row["volume"] == 1000 * sum(range(1, 32))  # 1到31的和

    def test_resample_to_month_multiple_assets(self):
        """测试多资产重采样"""
        dates = pd.date_range("2023-01-01", "2023-01-31", freq="D")
        data = []
        for date in dates[:5]:  # 只取前5天
            for asset in ["A", "B"]:
                data.append(
                    {
                        "date": date,
                        "asset": asset,
                        "close": 100 + date.day,
                        "volume": 1000 * date.day,
                    }
                )

        df = pd.DataFrame(data)
        result = resample_to_month(df, close="last", volume="sum")

        assert len(result) == 2  # 两个资产
        assert sorted(result.index.get_level_values("asset").unique()) == ["A", "B"]

    def test_resample_to_month_unsupported_column(self, sample_data):
        """测试不存在的列"""
        with pytest.raises(ValueError) as exc_info:
            resample_to_month(sample_data, nonexistent="last")
        assert "数据中不存在列" in str(exc_info.value)

    def test_resample_to_month_unsupported_method(self, sample_data):
        """测试不支持的聚合方法"""
        with pytest.raises(ValueError) as exc_info:
            resample_to_month(sample_data, close="median")  # median不支持
        assert "不支持的聚合方式" in str(exc_info.value)

    def test_resample_to_month_no_aggregation(self, sample_data):
        """测试没有指定聚合方法"""
        with pytest.raises(ValueError) as exc_info:
            resample_to_month(sample_data)  # 没有指定任何聚合
        assert "至少需要指定一个列的聚合方式" in str(exc_info.value)

    def test_resample_to_month_mean_aggregation(self, sample_data):
        """测试mean聚合方法"""
        result = resample_to_month(sample_data, close="mean")

        row = result.iloc[0]
        expected_mean = np.mean([105 + i for i in range(1, 32)])  # 1到31天的close值
        assert row["close"] == pytest.approx(expected_mean)

    def test_resample_to_month_unordered_data(self):
        """测试无序数据"""
        dates = pd.to_datetime(["2023-01-15", "2023-01-01", "2023-01-31"])
        data = []
        for date in dates:
            data.append({"date": date, "asset": "A", "close": date.day * 10})

        df = pd.DataFrame(data)
        result = resample_to_month(df, close="first")

        # 应该按日期排序后取第一个
        assert result.iloc[0]["close"] == 10  # 2023-01-01

        result = resample_to_month(df, close="last")
        # 应该按日期排序后取最后一个
        assert result.iloc[0]["close"] == 310  # 2023-01-31


class TestQfqAdjustment:
    """测试qfq_adjustment函数"""

    @pytest.fixture
    def stock_data(self):
        """创建股票数据，包含复权因子"""
        dates = pd.date_range("2023-01-01", "2023-01-05", freq="D")
        data = []

        # 模拟一个股票拆分：1月3日进行1拆2，复权因子从1变为0.5
        adj_factors = [1.0, 1.0, 0.5, 0.5, 0.5]

        for i, date in enumerate(dates):
            factor = adj_factors[i]
            base_price = 100 + i * 2  # 基础价格递增

            data.append(
                {
                    "date": date,
                    "asset": "A",
                    "open": base_price,
                    "high": base_price + 2,
                    "low": base_price - 2,
                    "close": base_price,
                    "volume": 10000 + i * 1000,
                    "adjust": factor,
                }
            )

        return pd.DataFrame(data)

    def test_qfq_adjustment_basic(self, stock_data):
        """测试基本前复权功能"""
        result = qfq_adjustment(stock_data, adj_factor_col="adjust")

        # 验证结果结构
        assert len(result) == len(stock_data)
        assert list(result.columns) == [
            "date",
            "asset",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "adjust",
        ]

        # 验证复权计算：以最新价格为基准
        # 最新日期的价格应该不变
        latest_original = stock_data.iloc[-1]
        latest_adjusted = result.iloc[-1]

        assert latest_adjusted["open"] == latest_original["open"]
        assert latest_adjusted["close"] == latest_original["close"]
        assert latest_adjusted["high"] == latest_original["high"]
        assert latest_adjusted["low"] == latest_original["low"]

        # 早期日期的价格应该被调整
        early_original = stock_data.iloc[0]
        early_adjusted = result.iloc[0]

        # 前复权：价格应该乘以复权因子比例
        expected_open = (
            early_original["open"]
            * early_original["adjust"]
            / latest_original["adjust"]
        )
        assert early_adjusted["open"] == expected_open

        # 成交量应该反向调整
        expected_volume = (
            early_original["volume"]
            * latest_original["adjust"]
            / early_original["adjust"]
        )
        assert early_adjusted["volume"] == expected_volume

    def test_qfq_adjustment_multiple_assets(self):
        """测试多资产前复权"""
        data = []

        # 资产A：有拆分
        for i, date in enumerate(pd.date_range("2023-01-01", "2023-01-03", freq="D")):
            data.append(
                {
                    "date": date,
                    "asset": "A",
                    "open": 100 + i * 5,
                    "high": 102 + i * 5,
                    "low": 98 + i * 5,
                    "close": 100 + i * 5,
                    "volume": 10000,
                    "adjust": [1.0, 1.0, 0.5][i],  # 第三天拆分
                }
            )

        # 资产B：无拆分
        for i, date in enumerate(pd.date_range("2023-01-01", "2023-01-03", freq="D")):
            data.append(
                {
                    "date": date,
                    "asset": "B",
                    "open": 50 + i * 2,
                    "high": 52 + i * 2,
                    "low": 48 + i * 2,
                    "close": 50 + i * 2,
                    "volume": 5000,
                    "adjust": 1.0,  # 无拆分
                }
            )

        df = pd.DataFrame(data)
        result = qfq_adjustment(df, adj_factor_col="adjust")

        # 验证每个资产独立处理
        asset_a_data = result[result["asset"] == "A"]
        asset_b_data = result[result["asset"] == "B"]

        # 资产A的价格应该被调整
        assert asset_a_data.iloc[0]["open"] != 100  # 应该被调整
        # 资产B的价格不应该变化（因为没有拆分）
        assert asset_b_data.iloc[0]["open"] == 50
        assert asset_b_data.iloc[1]["open"] == 52
        assert asset_b_data.iloc[2]["open"] == 54

    def test_qfq_adjustment_custom_column_name(self):
        """测试自定义复权因子列名"""
        data = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", "2023-01-03", freq="D"),
                "asset": "A",
                "open": [100, 102, 104],
                "high": [102, 104, 106],
                "low": [98, 100, 102],
                "close": [100, 102, 104],
                "volume": [10000, 11000, 12000],
                "custom_adj": [1.0, 1.0, 2],
            }
        )

        result = qfq_adjustment(data, adj_factor_col="custom_adj")

        # 验证使用自定义列名进行复权
        assert result.iloc[0]["open"] == 50.0

    def test_qfq_adjustment_empty_dataframe(self):
        """测试空DataFrame"""
        empty_df = pd.DataFrame(
            columns=[
                "date",
                "asset",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "adjust",
            ]
        )
        result = qfq_adjustment(empty_df, adj_factor_col="adjust")

        assert len(result) == 0
        assert list(result.columns) == [
            "date",
            "asset",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "adjust",
        ]
