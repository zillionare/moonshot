# moonshot


<p align="center">
<a href="https://pypi.python.org/pypi/moonshot">
    <img src="https://img.shields.io/pypi/v/moonshot.svg"
        alt = "Release Status">
</a>

<a href="https://github.com/zillionare/moonshot/actions">
    <img src="https://github.com/zillionare/moonshot/actions/workflows/main.yml/badge.svg?branch=release" alt="CI Status">
</a>

<a href="https://zillionare.github.io/moonshot/">
    <img src="https://img.shields.io/website/https/zillionare.github.io/moonshot/index.html.svg?label=docs&down_message=unavailable&up_message=available" alt="Documentation Status">
</a>

</p>


Alphalens lacks accurate monthly factor testing; our library delivers it.

## Overview

Moonshot is a lightweight, efficient Python package designed specifically for monthly factor-based investment strategy backtesting. **Unlike existing tools such as Alphalens, which has limitations in monthly factor analysis**, Moonshot provides a clean, intuitive API for evaluating factor performance through monthly portfolio rebalancing, making it ideal for quantitative researchers and portfolio managers working with factor investing strategies.

### Why Moonshot?

**Addressing Alphalens Limitations**: While Alphalens is a popular choice for factor analysis, it has known issues with monthly factor evaluation, particularly in handling monthly rebalancing logic and date alignment. Moonshot was specifically designed to address these shortcomings.

Unlike general-purpose backtesting frameworks, Moonshot is specifically crafted for monthly factor strategies. It eliminates the complexity of configuring generic backtesting engines while providing the precision needed for factor research. **Most importantly, it ensures accurate monthly factor analysis where other tools fall short.**

The name "Moonshot" reflects our ambition to help researchers and practitioners aim high in their factor investing endeavors - taking calculated risks for potentially significant returns, with the confidence that comes from accurate analysis.

* Free software: MIT
* Documentation: <https://zillionare.github.io/moonshot/>


## Installation

### From PyPI (Recommended)

```bash
pip install moonshot
```

### From Source

```bash
git clone https://github.com/zillionare/moonshot.git
cd moonshot
pip install -e .
```

## Quick Start

### Basic Usage

```python
import pandas as pd
import numpy as np
from moonshot import monthly_factor_backtest, calculate_group_statistics, plot_cumulative_returns

# Create simple synthetic data for demonstration
# Factor data: higher factor values should lead to higher returns
factor_data = pd.Series([
    1.0, 2.0, 3.0,  # January 31st: A=1.0, B=2.0, C=3.0
    1.5, 2.5, 3.5   # February 28th: A=1.5, B=2.5, C=3.5
], index=pd.MultiIndex.from_tuples([
    ('2023-01-31', 'A'), ('2023-01-31', 'B'), ('2023-01-31', 'C'),
    ('2023-02-28', 'A'), ('2023-02-28', 'B'), ('2023-02-28', 'C')
], names=['date', 'asset']), name='factor')

# Price data: Include both month-end (for factor calculation) and next month start (for trading)
price_data = pd.DataFrame([
    # January month-end data (factor calculation date)
    {'date': '2023-01-31', 'asset': 'A', 'open': 100, 'close': 100},
    {'date': '2023-01-31', 'asset': 'B', 'open': 100, 'close': 100},
    {'date': '2023-01-31', 'asset': 'C', 'open': 100, 'close': 100},
    # February start data (trading execution date)
    {'date': '2023-02-01', 'asset': 'A', 'open': 100, 'close': 100},
    {'date': '2023-02-01', 'asset': 'B', 'open': 100, 'close': 100},
    {'date': '2023-02-01', 'asset': 'C', 'open': 100, 'close': 100},
    # February month-end data (factor calculation date)
    {'date': '2023-02-28', 'asset': 'A', 'open': 100, 'close': 100},  # 0% return
    {'date': '2023-02-28', 'asset': 'B', 'open': 100, 'close': 110},  # 10% return
    {'date': '2023-02-28', 'asset': 'C', 'open': 100, 'close': 120},  # 20% return
    # March start data (trading execution date)
    {'date': '2023-03-01', 'asset': 'A', 'open': 100, 'close': 100},
    {'date': '2023-03-01', 'asset': 'B', 'open': 110, 'close': 110},
    {'date': '2023-03-01', 'asset': 'C', 'open': 120, 'close': 120},
]).set_index(['date', 'asset'])

# Run backtest with 3 quantiles
strategy_returns, benchmark_returns = monthly_factor_backtest(
    factor_data, price_data, quantiles=3
)

print("Strategy Returns by Quantile:")
print(strategy_returns)
# Expected output: Q3 (highest factor) should have ~20% return
# Q2 (medium factor) should have ~10% return
# Q1 (lowest factor) should have ~0% return

print("\nBenchmark Returns (Equal-weighted):")
print(benchmark_returns)
# Expected output: ~10% (average of 0%, 10%, 20%)
```

### Advanced Analysis

```python
# Calculate performance statistics
stats = calculate_group_statistics(strategy_returns)
print("\nPerformance Statistics:")
print(stats)

# Plot cumulative returns
plot_cumulative_returns(strategy_returns, benchmark_returns)

# Analyze long-short spread
from moonshot import analyze_long_short_spread
spread = analyze_long_short_spread(strategy_returns)
print("\nLong-Short Spread:")
print(spread)
```

### Custom Binning

```python
# Use custom bins instead of quantiles
strategy_returns, benchmark_returns = monthly_factor_backtest(
    factor_data, price_data, bins=[0, 1.5, 2.5, 4.0]
)
```

## Examples

For more comprehensive examples and tutorials, see the [`examples/`](examples/) directory:

- **[`basic_example.py`](examples/basic_example.py)**: Introduction to core functionality
- **[`advanced_analysis.py`](examples/advanced_analysis.py)**: Performance analysis and visualization
- **[`custom_binning.py`](examples/custom_binning.py)**: Quantiles vs custom bins comparison
- **[`data_preprocessing.py`](examples/data_preprocessing.py)**: Handling real-world data challenges

Run any example with:
```bash
python examples/basic_example.py
```

## Important Data Requirements

### ⚠️ Critical: Date Alignment

**Moonshot assumes your data is properly aligned to month-end dates.** The library does not automatically handle missing dates or irregular time series. If your data has gaps or misaligned dates, you **must** preprocess it before using Moonshot.

### Data Preprocessing for Missing Dates

If you need to trade even when some dates are missing, align your data to month-end dates first:

```python
# Method 1: Forward fill with complete date range
def align_to_month_end(data, start_date, end_date):
    """
    Align data to month-end dates using forward fill
    """
    # Generate complete month-end date range
    complete_dates = pd.date_range(start_date, end_date, freq='ME')

    if isinstance(data, pd.Series):
        # For factor data (MultiIndex with date, asset)
        assets = data.index.get_level_values('asset').unique()
        complete_index = pd.MultiIndex.from_product(
            [complete_dates, assets], names=['date', 'asset']
        )
        aligned_data = data.reindex(complete_index).groupby('asset').ffill()
    else:
        # For price data (DataFrame with MultiIndex)
        assets = data.index.get_level_values('asset').unique()
        complete_index = pd.MultiIndex.from_product(
            [complete_dates, assets], names=['date', 'asset']
        )
        aligned_data = data.reindex(complete_index).groupby('asset').ffill()

    return aligned_data.dropna()

# Example usage
aligned_factor = align_to_month_end(factor_data, '2023-01-31', '2023-12-31')
aligned_prices = align_to_month_end(price_data, '2023-01-31', '2023-12-31')

# Method 2: Resample to month-end and forward fill
def resample_to_month_end(data):
    """
    Resample daily data to month-end
    """
    if isinstance(data, pd.Series):
        return data.groupby('asset').resample('ME', level='date').last().ffill()
    else:
        return data.groupby('asset').resample('ME', level='date').last().ffill()
```

### Data Format Requirements

1. **Factor Data**: Must be a pandas Series with MultiIndex (date, asset)
2. **Price Data**: Must be a pandas DataFrame with MultiIndex (date, asset) and columns ['open', 'close']
3. **Dates**: Must be pandas datetime objects, preferably month-end dates
4. **No Missing Values**: Ensure no NaN values in critical periods

## API Reference

### Core Functions

- `monthly_factor_backtest(factor_data, bars_data, quantiles=None, bins=None)`: Main backtesting function
- `calculate_group_statistics(monthly_returns)`: Calculate performance metrics
- `plot_cumulative_returns(strategy_returns, benchmark_returns)`: Visualize results
- `analyze_long_short_spread(monthly_returns)`: Analyze long-short strategy

For detailed API documentation, visit: <https://zillionare.github.io/moonshot/>

## Features

* **Accurate Monthly Factor Analysis**: Specifically designed to handle monthly rebalancing correctly
* **Simple API**: Clean, intuitive interface for factor backtesting
* **Flexible Grouping**: Support both quantile-based and custom bin-based grouping
* **Performance Analytics**: Built-in calculation of key performance metrics
* **Visualization**: Easy plotting of cumulative returns and performance
* **Long-Short Analysis**: Dedicated tools for long-short strategy evaluation
* **Type Safety**: Full type hints for better development experience

## Credits

This package was created with the [ppw](https://zillionare.github.io/python-project-wizard) tool. For more information, please visit the [project page](https://zillionare.github.io/python-project-wizard/).
