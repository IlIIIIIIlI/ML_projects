https://thecleverprogrammer.com/2021/12/27/cryptocurrency-price-prediction-with-machine-learning/

1. **数据下载与准备**：使用`yf.download`方法下载比特币历史价格数据。指定起始和结束日期，聚焦于最近两年的数据。

```python
import pandas as pd
import yfinance as yf
from datetime import date, timedelta

# 设置日期范围
today = date.today()
end_date = today.strftime("%Y-%m-%d")
start_date = (today - timedelta(days=730)).strftime("%Y-%m-%d")

# 从yfinance库下载数据
data = yf.download('BTC-USD', start=start_date, end=end_date, progress=False)

# 数据处理
data["Date"] = data.index
data = data[["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]]
```

2. **数据可视化**：通过`plotly.graph_objects`，我们能创建一个交互式的蜡烛图，显示比特币的开盘、收盘、最高和最低价格。

```python
figure = go.Figure(data=[go.Candlestick(x=data["Date"],
                                        open=data["Open"], 
                                        high=data["High"],
                                        low=data["Low"], 
                                        close=data["Close"])])
```

- **数据中的特征**：在这个数据集中，有“Open”、“High”、“Low”、“Volume”等作为输入特征，输出是“Close”。与股票价格预测类似，每天的加密货币价格可能受到前几天价格的影响。

```Python
from autots import AutoTS

# 使用AutoTS库进行预测
model = AutoTS(forecast_length=30, frequency='infer', ensemble='simple')
model = model.fit(data, date_col='Date', value_col='Close', id_col=None)
forecast = model.predict().forecast
```

- **AutoTS()**：是该库的核心类。`forecast_length`参数表示预测的长度，而`frequency`和`ensemble`是其他相关参数，影响模型的选择和集成方法。
- **model.fit()**：用于训练模型。其中`date_col`指定日期列，`value_col`指定我们要预测的列。
- **model.predict()**：进行预测并返回结果。



数字货币价格预测模型采用了时间序列分析的AutoTS库。此库专为时间序列预测而设计，具有自动选择最佳模型、处理单变量和多变量时间序列、自动处理缺失值和异常值等功能。

**<u>功能</u>**

- **自动模型选择**：AutoTS可以评估多种时间序列模型（如ARIMA、SARIMA、Prophet等），并根据预测性能自动选择最佳模型。这消除了人工选择模型的需要，节省了大量时间。根据所处理的数据类型，它能自动找到最合适的时间序列预测模型。支持多变量输出，概率预测，并且大部分模型可以轻松地扩展到数十万甚至数百万的输入系列。此外，它还具有一个 AutoML 特性搜索功能，该功能通过遗传算法自动查找给定数据集的最佳模型、预处理和集成。

- **处理单变量和多变量时间序列**：不仅可以分析一个因变量随时间变化的模式，还可以同时分析多个相关的变量。例如，在加密货币的情况下，可能不仅考虑价格，还要考虑交易量、市场情绪等其他因素。

- **数据清洗**：AutoTS能够处理“脏数据”，即缺失值和异常值。这对于时间序列分析至关重要，因为缺失的观察结果或极端值可能严重扭曲预测。数据中可能存在缺失或异常的价格，AutoTS库能自动处理这些情况，确保模型的稳定性。

- **时间序列转换**：AutoTS 提供了30多种特定于时间序列的转换，这些转换也是按照 sklearn 的风格设计的。转换是为了将时间序列数据转换为更适合模型学习的形式。


- 它的核心是一个自动化的机器学习流程，用于选择最佳的模型和预处理方法来解决特定的时间序列预测问题。

<u>**模型选择和评估**</u>
AutoTS 包括多种时间序列预测模型，如朴素方法、统计学方法、机器学习和深度学习模型。每个模型都有其特定的假设和应用场景。AutoTS 使用交叉验证和其他评估技术来比较这些模型在给定数据集上的性能。

数学上，模型评估可以通过各种损失函数（如 MAE, RMSE 等）来量化，这些损失函数计算预测值和实际值之间的差异：
$
\text{MAE} = \frac{\sum_{i=1}^{n}\left|y_i - \hat{y_i}\right|}{n}
$
其中 $y_i$ 是实际值，$\hat{y_i}$ 是预测值，$n$ 是观测值的数量。

1. **遗传算法用于特征选择和模型调优**:
   遗传算法是一种启发式优化技术，它模拟自然选择过程来选择最佳的模型参数和特征。在 AutoTS 中，它被用来自动找到最佳模型、预处理、参数、特征和集成方法。

   算法的每一代都会生成一组候选解决方案（即模型配置），并使用适应性函数（基于预测性能的评分）来评估它们。最佳表现的“个体”会被选中并“交叉”产生下一代的候选解决方案。这个过程一直持续到满足特定的停止准则（如达到一定的准确性或迭代次数）。

   这一过程可以通过以下伪代码来简化描述：

   ```
   初始化种群
   while (not 满足终止条件) {
       评估种群中每个个体的适应度
       选择最适应的个体
       进行交叉和变异以产生新一代的种群
   }
   返回最佳解决方案
   ```

2. **集成方法**:
   AutoTS 使用了水平和马赛克式集成，这是一种高级的集成技术，可以结合多个模型的预测来提高准确性。基本思想是，不同的模型可能在不同方面表现良好，通过集成，我们可以平衡这些优点，减少单个模型可能的缺点。

   集成的计算可以视为加权平均，其中每个模型的预测都根据其在验证数据上的性能进行加权。如果 $m$ 是模型数量，$w_i$ 是第 $i$ 个模型的权重（基于其性能），则集成预测可以表示为：
   $
   \hat{y} = \sum_{i=1}^{m} w_i \cdot \hat{y_i}
   $
   其中 $\hat{y_i}$ 是第 $i$ 个模型的预测。

**<u>基本使用方法</u>**

**a. 导入库和数据**：

```python
from autots import AutoTS, load_daily
df = load_daily(long=long)
```
**b. 定义模型**：
```python
model = AutoTS(
    forecast_length=21,
    ...
    validation_method="backwards"
)
```
**c. 训练模型**：
```python
model = model.fit(
    df,
    date_col='datetime' if long else None,
    ...
    id_col='series_id' if long else None,
)
```
**d. 预测**：
```python
prediction = model.predict()
prediction.plot(...)
print(model)
forecasts_df = prediction.forecast
```
ta还提供了许多用于加速和处理大数据的技巧，例如使用子集参数、调整 n_jobs 参数、使用结果文件方法保存进度等。
