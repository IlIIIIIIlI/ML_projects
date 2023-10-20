https://thecleverprogrammer.com/2022/01/17/time-series-analysis-using-python/

![image-20231019000159791](https://photosavercn.oss-cn-guangzhou.aliyuncs.com/img/202310190002840.png)

定义日期范围

```python
today = date.today()
# 获取当前日期，并设置为字符串格式，作为结束日期。
d1 = today.strftime("%Y-%m-%d")
end_date = d1
# 计算720天前的日期，并设置为字符串格式，作为开始日期。
d2 = date.today() - timedelta(days=720)
d2 = d2.strftime("%Y-%m-%d")
start_date = d2
```

- 定义起始和结束日期以下载股票数据。
- 使用`timedelta`定义过去的720天作为起始日期。

**<u>下载股票数据</u>**

```python
data = yf.download('AAPL', 
                      start=start_date, 
                      end=end_date, 
                      progress=False)
print(data.head())
```

- 使用`yfinance`下载苹果公司(AAPL)的股票数据。
- 打印前五条数据进行查看。

**<u>使用Plotly绘制图形</u>**

```python
import plotly.express as px
```

- `plotly.express`: 一个用于创建交互式图表的库。

```python
# 创建一个线性图表，显示收盘价随时间的变化。
figure = px.line(data, x = data.index, 
                 y = "Close", 
                 title = "Time Series Analysis (Line Plot)")
figure.show()
```

- 创建时间序列的折线图，显示了在给定期间内股票的收盘价。这有助于我们快速识别价格趋势和模式。

```python
figure = go.Figure(data=[go.Candlestick(x = data.index,
                                        open = data["Open"], 
                                        high = data["High"],
                                        low = data["Low"], 
                                        close = data["Close"])])
figure.update_layout(title = "Time Series Analysis (Candlestick Chart)", 
                     xaxis_rangeslider_visible = False)
figure.show()
```

- 创建蜡烛图，显示开市、最高、最低和闭市价格。

```python
figure = px.bar(data, x = data.index, y = "Close", title = "Time Series Analysis (Bar Plot)")
figure.show()
```

- 创建时间序列的柱状图，显示了收盘价的变化。与线图相比，条形图能更明确地展示价格的变化幅度。

```python
figure = px.line(data, x = data.index, 
                 y = 'Close', 
                 range_x = ['2021-07-01','2021-12-31'], 
                 title = "Time Series Analysis (Custom Date Range)")
figure.show()
```

- 创建限定日期范围的折线图。允许我们指定一个自定义的日期范围。研究特定时间内的价格动态时非常有用

```python
# 创建一个带有操作按钮和滑动条的K线图表。
figure = go.Figure(data = [go.Candlestick(x = data.index,
                                          open = data["Open"], 
                                          high = data["High"],
                                          low = data["Low"], 
                                          close = data["Close"])])
figure.update_layout(title = "Time Series Analysis (Candlestick Chart with Buttons and Slider)")
#  添加时间范围选择器
figure.update_xaxes(
    rangeslider_visible = True,
    rangeselector = dict(
        buttons = list([
            dict(count = 1, label = "1m", step = "month", stepmode = "backward"),
            dict(count = 6, label = "6m", step = "month", stepmode = "backward"),
            dict(count = 1, label = "YTD", step = "year", stepmode = "todate"),
            dict(count = 1, label = "1y", step = "year", stepmode = "backward"),
            dict(step = "all")
        ])))
figure.show()
```

- 增强了K线图表，包括添加了可让用户通过点击选择不同时间范围的按钮和滑动条。这提供了一个交互式工具，使分析师能够轻松查看不同时间段的价格变化。

对于本代码中的股票数据，通过不同的图表，如折线图、蜡烛图和柱状图，我们可以轻松地观察到股票价格的变化趋势、波动情况和关键日期的数据点。此外，交互式图表如带有按钮和滑块的图表，可以让用户自定义查看的日期范围，增强了数据分析的灵活性。
