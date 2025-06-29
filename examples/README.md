# Moonshot Examples

This directory contains practical examples demonstrating various features and use cases of the Moonshot library.

## 示例文件

本目录包含Python脚本和Jupyter Notebook两种格式的示例，方便不同使用场景：

### 1. 基础示例 (`basic_example`)
**用途**: 演示Moonshot的基本用法
**内容**:
- 创建简单的合成数据
- 运行基础的因子回测
- 理解输入数据格式和输出结果

**运行方式**:
```bash
# Python脚本
python basic_example.py

# Jupyter Notebook
jupyter notebook basic_example.ipynb
```

### 2. 高级分析 (`advanced_analysis`)
**用途**: 展示高级分析功能
**内容**:
- 性能统计计算
- 累计收益可视化
- 多空价差分析
- 使用更长时间序列的数据

**运行方式**:
```bash
# Python脚本
python advanced_analysis.py

# Jupyter Notebook
jupyter notebook advanced_analysis.ipynb
```

### 3. 自定义分箱 (`custom_binning`)
**用途**: 演示自定义分箱功能
**内容**:
- 对比分位数分箱与自定义分箱
- 展示不同分箱方法的效果
- 分析分箱方法的选择原则

**运行方式**:
```bash
# Python脚本
python custom_binning.py

# Jupyter Notebook
jupyter notebook custom_binning.ipynb
```

### 4. 数据预处理 (`data_preprocessing`)
**用途**: 数据预处理示例
**内容**:
- 处理缺失日期的数据
- 两种数据对齐方法
- 数据预处理的最佳实践

**运行方式**:
```bash
# Python脚本
python data_preprocessing.py

# Jupyter Notebook
jupyter notebook data_preprocessing.ipynb
```

## 运行示例

### 环境要求
确保已安装Moonshot及其依赖：
```bash
pip install moonshot
```

### 运行所有示例
按顺序运行所有Python脚本示例：
```bash
# 从项目根目录执行
for script in examples/*.py; do
    echo "正在运行 $script..."
    python "$script"
    echo "完成 $script"
    echo "---"
done
```

### 单独执行
运行特定示例：
```bash
# Python脚本
python examples/basic_example.py
python examples/advanced_analysis.py
python examples/custom_binning.py
python examples/data_preprocessing.py

# Jupyter Notebook
jupyter notebook examples/basic_example.ipynb
jupyter notebook examples/advanced_analysis.ipynb
jupyter notebook examples/custom_binning.ipynb
jupyter notebook examples/data_preprocessing.ipynb
```

## 数据结构约定

所有示例都遵循以下数据结构约定：

### 因子数据
- **类型**: `pd.Series`
- **索引**: `pd.MultiIndex`，层级为 `['date', 'asset']`
- **值**: 数值型因子值
- **名称**: 字符串（如 'factor'）

### 价格数据
- **类型**: `pd.DataFrame`
- **索引**: `pd.MultiIndex`，层级为 `['date', 'asset']`
- **列**: `['open', 'close']`（最低要求）
- **值**: 数值型价格数据

### 日期要求
- 因子计算日期（通常为月末）
- 交易执行日期（通常为次月初）
- 因子数据和价格数据的日期必须对齐

## 常用模式

### 1. 数据创建模式
```python
# 创建因子数据
factor_data = pd.Series(
    values,
    index=pd.MultiIndex.from_tuples(index_tuples, names=['date', 'asset']),
    name='factor'
)

# 创建价格数据
price_data = pd.DataFrame(records).set_index(['date', 'asset'])
```

### 2. 回测执行模式
```python
# 运行回测
strategy_returns, benchmark_returns = monthly_factor_backtest(
    factor_data, price_data, quantiles=5
)

# 分析结果
stats = calculate_group_statistics(strategy_returns)
fig = plot_cumulative_returns(strategy_returns, benchmark_returns)
spread_analysis = analyze_long_short_spread(strategy_returns)
```

### 3. 数据预处理模式
```python
# 将数据对齐到月末
aligned_factor = align_to_month_end(factor_data, start_date, end_date)
aligned_price = resample_to_month_end(price_data, start_date, end_date)

# 验证对齐
assert aligned_factor.index.get_level_values('date').equals(
    aligned_price.index.get_level_values('date')
)
```

## 故障排除

### 常见问题

1. **日期对齐错误**
   - **问题**: 因子数据和价格数据的日期不匹配
   - **解决方案**: 使用数据预处理函数对齐日期
   - **参考**: 查看 `data_preprocessing.py`

2. **缺少交易日期**
   - **问题**: 只提供了月末日期，没有交易执行日期
   - **解决方案**: 在价格数据中添加次月初日期
   - **参考**: 所有示例都展示了正确的日期结构

3. **空组合**
   - **问题**: 自定义分箱导致某些分位数组合为空
   - **解决方案**: 调整分箱边界或使用分位数分箱
   - **参考**: 查看 `custom_binning.py` 的对比

4. **数据类型问题**
   - **问题**: 非数值数据或错误的索引类型
   - **解决方案**: 确保正确的数据类型和MultiIndex结构
   - **参考**: 所有示例都展示了正确的数据格式

### 获取帮助

如果遇到问题：
1. 检查上述数据结构要求
2. 查看相关用例的示例
3. 确保所有日期都正确对齐
4. 验证数据是否符合预期格式

## 下一步

运行这些示例后：
1. **适配您的数据**: 修改数据创建函数以适用于您的实际数据
2. **实验参数**: 尝试不同的分位数、分箱和分析周期
3. **添加自定义分析**: 用您自己的性能指标扩展示例
4. **扩大规模**: 将模式应用于更大的数据集和更长的时间周期

更高级的用法请参考主文档和API参考。
