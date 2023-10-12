https://thecleverprogrammer.com/2022/01/03/stock-price-prediction-with-lstm/

想象一下，你正在看一部很长的电影，而 LSTM 就像你的大脑一样。在观看的过程中，你的大脑会不自觉的记住某些关键场景、角色和情节，并在观影过程中不断地回忆和参照这些信息来理解电影的故事。LSTM 做的事情也类似：它在处理数据序列（例如股票价格）时，会记住过去的信息，并利用这些信息来理解或预测未来的走势。

1. **时间依赖性：** LSTM（长短时记忆）网络层适用于时间序列数据，如股票价格，因为它能“记住”过去的信息。它们能够捕捉到数据中的时间依赖性，这对预测未来价格至关重要。
2. **避免梯度消失/爆炸问题：** 在处理长序列时，普通的循环神经网络(RNN)常常遇到梯度消失或爆炸的问题。LSTM 通过其特殊的结构帮助缓解这些问题，使其更适合处理时间序列问题。

```Python
from keras.models import Sequential
from keras.layers import Dense, LSTM
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape= (xtrain.shape[1], 1)))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))
model.summary()
```

- 数据中有四个特征（Open、High、Low、Volume）作为输入，而输出是Close。由于每天的价格可能受到前几天的价格影响，使用LSTM层是合适的，因为它能够捕获这种时间序列数据中的长期依赖关系。
- 模型中**第一个LSTM层**的选择为128个单元; 一般而言，第一层通常比第二层有更多的单元，以捕获更多的信息。`return_sequences=True` 表示这一层输出的每一个时间步的隐藏状态都会被传递到下一层。这对堆叠 LSTM 层是必要的。
- **第二个LSTM层选择为64个单元**, 这层不返回整个序列到下一层，因为后面接的是全连接层，它期望的输入是单个样本，而不是序列。
- **之后的Dense层**是为了进行特征转换和输出预测。这个层常常被用作隐藏层在输出层之前捕获更多的<u>非线性特征。</u>
- **输出层(1个单元)**：我们的目标是预测下一个收盘价，所以我们只需要1个节点来输出预测值。

在这个例子中，我们只用到了"Open", "High", "Low", "Volume"来作为输入特征，可能的原因包括：

- **模型输入**：在上文的模型训练中，输入数据 `x` 是由 "Open", "High", "Low", "Volume" 构成的。这些特征包含了当天的股票动态，可能被认为对于预测收盘价“Close”是有用的。
- **避免数据泄露**：我们没有使用“Close”和“Adj Close”作为输入特征，可能是为了避免未来数据的泄露。理论上，我们在预测未来的收盘价时，不应该知道未来的收盘价。

**数据下载**

- `yf.download`: 通过`yfinance`库下载股票数据。

**日期和时间处理**
- `date.today()`: 获取今天的日期。
- `strftime`: 格式化日期为字符串。
- `timedelta`: 表示两个日期或时间之间的差异。

**数据处理**

- `data["Date"] = data.index`: 将日期从index列移到新的"Date"列。
- `data[["Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"]]`: 选择数据的子集。

**数据可视化**
- `go.Figure`: 创建一个Plotly图形对象。
- `go.Candlestick`: 创建一个蜡烛图对象。

**数据分析**
- `data.corr()`: 计算DataFrame中列与列之间的相关系数。

**构建和训练神经网络模型**
- `Sequential`: 一个线性堆叠的层模型，用于创建神经网络。
- `model.summary()`: 输出模型的摘要信息，包括层数、每层的输出维度等。
- `model.compile`: 配置模型的学习过程，包括指定优化器和损失函数等。
- `model.fit`: 在训练集上训练模型。





![image-20231012020221196](https://photosavercn.oss-cn-guangzhou.aliyuncs.com/img/202310120202222.png)

在LSTM中，每一个门（遗忘门、输入门、输出门）和单元状态的更新都有自己的权重矩阵和偏置向量。具体来说，给定一个输入特征数量 $D$ 和隐藏单元数量 $H$，我们有以下参数计算方式：

- **权重矩阵**: 对于每个门有一个权重矩阵，大小为 $[D, H]$；还有用于单元状态的权重矩阵，大小同样是 $[D, H]$。一共有4个这样的矩阵，所以权重参数总数为 $4 \times D \times H$。
  
- **循环权重矩阵**: 对于每个门和单元状态更新，我们还有一个循环权重矩阵，大小为 $[H, H]$。所以这些参数加起来是 $4 \times H \times H$。

- **偏置向量**: 每个门和单元状态更新有一个偏置向量，长度为 $H$。总共 $4 \times H$ 个偏置参数。

综合上述参数，一个LSTM层的总参数数量为:
$ 
\text{Total LSTM params} = 4 \times (D \times H + H^2 + H)
$

- 第一层LSTM:
  - $D = 1$ (因为输入特征数为1)
  - $H = 128$
  - $
  \text{Total params for first LSTM layer} = 4 \times (1 \times 128 + 128^2 + 128) = 66560
  $

- 第二层LSTM (注意，现在D取决于上一层的H):
  - $D = 128$ 
  - $H = 64$
  - $
  \text{Total params for second LSTM layer} = 4 \times (128 \times 64 + 64^2 + 64) = 49408
  $

Dense layer的参数计算相对简单。给定输入节点数 $N$ 和输出节点数 $M$，权重参数数量是 $N \times M$，偏置参数数量是 $M$。即:
$ 
\text{Total Dense params} = N \times M + M
$

- 第一个Dense层:
  - $N = 64$ 
  - $M = 25$
  - $
  64 \times 25 + 25 = 1625
  $

- 第二个Dense层:
  - $N = 25$ 
  - $M = 1$
  - $
  25 \times 1 + 1 = 26
  $

整个模型的参数总量：$ 66560 + 49408 + 1625 + 26 = 117619 $

像股票价格预测这样的任务，过多的参数有时会捕获到数据中的噪声（而不是真正的趋势），因此可能导致过拟合。

考虑到有117,619个参数，需确保你有足够多的训练数据。一般来说，希望每个参数能至少对应到几个样本点，以避免过拟合。



![image-20231012020239964](https://photosavercn.oss-cn-guangzhou.aliyuncs.com/img/202310120202987.png)

![image-20231012020042341](https://photosavercn.oss-cn-guangzhou.aliyuncs.com/img/202310120200398.png)