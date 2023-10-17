https://thecleverprogrammer.com/2022/02/01/waiter-tips-prediction-with-machine-learning/

在餐厅环境中，小费的分配往往受到多种因素的影响，包括账单总额、客户数量、星期几等多个变量。我们将通过一系列数据科学技巧深入探讨这些因素，同时借助机器学习模型预测小费的分配。

**<u>数据集分析</u>**

1. **总账单（total_bill）**：包括税款在内的整体账单金额。
2. **小费（tip）**：给予侍者的小费金额。
3. **性别（sex）**：付账者的性别。
4. **吸烟者（smoker）**：付账者是否为吸烟者。
5. **星期几（day）**：一周中的哪一天。
6. **时间（time）**：午餐或晚餐。
7. **人数（size）**：桌子上的人数。

**<u>数据预处理及编码</u>**

在模型训练之前，处理分类数据是关键。例如，性别(`sex`)、是否吸烟(`smoker`)、一周中的某一天(`day`)和时间(`time`)都是非数值特征。我们使用映射来将它们转化为数值型数据：

```python
data["sex"] = data["sex"].map({"Female": 0, "Male": 1})
data["smoker"] = data["smoker"].map({"No": 0, "Yes": 1})
data["day"] = data["day"].map({"Thur": 0, "Fri": 1, "Sat": 2, "Sun": 3})
data["time"] = data["time"].map({"Lunch": 0, "Dinner": 1})
```

**<u>作图</u>**

1. 使用`plotly.express`库中的`scatter`函数绘制散点图，用于探索总账单、小费和桌子大小之间的关系，同时以“day”变量对点进行着色，并拟合了一个OLS（普通最小二乘法）趋势线。

```python
figure = px.scatter(data_frame = data, x="total_bill",
                    y="tip", size="size", color= "sex", trendline="ols")
figure.show()
figure = px.scatter(data_frame = data, x="total_bill",
                    y="tip", size="size", color= "time", trendline="ols")
figure.show()
```

2. 第一个饼图关注一周中的每一天并展示了每一天小费的分布。这有助于理解哪些天的小费更为丰厚。第二个关注性别的分布，以理解男性或女性是否通常给出更多的小费。第三个展示了吸烟者和非吸烟者在小费上的分布，帮助分析是否吸烟与给出的小费之间存在关系。最后一个则观察了午餐和晚餐之间在小费分布上的差异，提供了用餐时间与小费是否有关的线索。

```python
figure = px.pie(data, values='tip', names='day',hole = 0.5)
figure.show()
figure = px.pie(data, values='tip', names='sex',hole = 0.5)
figure.show()
figure = px.pie(data, values='tip', names='smoker',hole = 0.5)
figure.show()
figure = px.pie(data, values='tip', names='time',hole = 0.5)
figure.show()
```

**线性回归模型**

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

x = np.array(data[["total_bill", "sex", "smoker", "day", "time", "size"]])
y = np.array(data["tip"])

# 数据分割
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2, random_state=42)

# 模型训练
model = LinearRegression()
model.fit(xtrain, ytrain)
```

![37251697262685_.pic](../../../../../Library/Containers/com.tencent.xinWeChat/Data/Library/Application Support/com.tencent.xinWeChat/2.0b4.0.9/ae5e2ba1bd24e1a647bf49b0e8f39752/Message/MessageTemp/ab505672c79e637f752fbb3126e4c2b8/Image/37251697262685_.pic.jpg)

Ridge 回归参数概述

- `alpha`: 控制正则化强度的参数。
- `fit_intercept`: 布尔值，是否计算此模型的截距。
- `solver`: 用于计算程序的解算器，包括‘auto’、‘svd’、‘cholesky’、‘lsqr’等。
- `max_iter`: 整数，共轭梯度解算器的最大迭代次数。
- `tol`: 浮点数，解（系数_）的精度。
- `positive`: 布尔值，当设置为True时，强制系数为正。

特点

- 内建支持多变量回归（即当 y 是形状为 (n_samples, n_targets) 的2d-array时）。
- 通过L2正则化来解决特征共线性的问题。
- $\alpha = 0$ 时，Ridge回归等价于普通最小二乘法（但不建议在 Ridge 对象中使用 $\alpha = 0$）。

数学推导

对于目标函数 $J(w)$，我们可以使用梯度下降（Gradient Descent）来找到使 $J(w)$ 最小化的 $w$。

计算梯度（Gradient）:
$
\frac{\partial J(w)}{\partial w} = -2X^T(y-Xw) + 2\alpha w
$

更新权重（Weight Update Rule）:
$
w := w - \eta \frac{\partial J(w)}{\partial w}
$
其中 $\eta$ 是学习率。

其他

- `fit(X, y)`: 拟合 Ridge 回归模型。
- `predict(X)`: 使用线性模型进行预测。
- `score(X, y)`: 返回预测的决定系数 R^2。

- 当特征数量大并且特征之间存在共线性时，Ridge 回归特别有效。
- $\alpha$ 的选择需要通过交叉验证来确定，可以使用 `RidgeCV` 自动选择。



Lasso

Lasso回归是一种估计稀疏系数的线性模型，因其倾向于寻找少量非零系数的解，从而实际上减少了依赖特征的数量，它在一些上下文中很有用。在某些条件下，Lasso能恢复精确的非零系数集。

目标函数为最小化如下：
$ \min_{w} \frac{1}{2n_{\text{samples}}} ||Xw - y||_2^2 + \alpha ||w||_1 $
其中，

- $ X $ 是输入数据
- $ w $ 是权重系数
- $ y $ 是输出数据
- $ ||w||_1 $ 是权重的L1范数
- $ \alpha $ 是控制稀疏性的正则化参数

该模型试图找到一个权重向量 $ w $，它最小化预测和实际输出之间的平方误差，同时保持权重的L1范数最小。

坐标下降法

Lasso类使用坐标下降法作为拟合系数的算法。对于每个特征坐标，我们解决一个单变量优化问题，保持其他坐标固定。这种方法在计算上是高效的，尤其是当输入特征数量远大于样本数量时。

结论

- 由于Lasso回归产生稀疏模型，因此可以用于特征选择。
- `alpha`参数控制估计系数的稀疏程度。
- 使用交叉验证可以自动选择最佳的`alpha`值。在`sklearn`中，可以使用LassoCV和LassoLarsCV，这两个对象通过交叉验证设置Lasso的alpha参数。对于具有许多共线特征的高维数据集，LassoCV通常更可取。然而，相对于LassoCV，LassoLarsCV有探索更多相关alpha参数值的优势，并且在样本数量相对于特征数量非常小的情况下通常比LassoCV更快。

总之，Lasso回归是一种稀疏模型，特别适合于特征选择和处理具有多重共线性的高维数据集。



**<u>ElasticNet 线性回归</u>**

ElasticNet 是一个结合了 L1 和 L2 作为正则化器的线性回归模型。其中，L1 正则化趋向于产生稀疏权重，L2 正则化能防止模型过拟合。ElasticNet 适用于多重共线性问题。

**<u>主要目标函数:</u>**

$ \text{minimize} \left\{ \frac{1}{2n_{\text{samples}}} ||y - Xw||^2_2 + \alpha \times \text{l1\_ratio} \times ||w||_1 + 0.5 \times \alpha \times (1 - \text{l1\_ratio}) \times ||w||^2_2 \right\} $

其中，$ w $ 是模型参数，$ X $ 是输入数据，$ y $ 是输出数据，$ ||\cdot||_2 $ 是 L2 范数，$ ||\cdot||_1 $ 是 L1 范数，$ \alpha $ 是正则化的强度，l1_ratio 是 L1 正则化与 L2 正则化之间的比率。

**<u>等价形式:</u>**

如果你对分别控制 L1 和 L2 惩罚感兴趣，可以注意到目标函数可以写成：

$ \text{minimize} \left\{ a \times ||w||_1 + 0.5 \times b \times ||w||^2_2 \right\} $

其中 $ \alpha = a + b $ 并且 $ \text{l1\_ratio} = \frac{a}{a + b} $

**<u>参数解释:</u>**

- `alpha` (float, default=1.0): 正则项的乘法因子。当 $\alpha = 0$ 时，模型等价于普通的最小二乘法。
  
- `l1_ratio` (float, default=0.5): 用于调整正则项中 L1 和 L2 的比例。

- `fit_intercept` (bool, default=True): 是否需要计算截距项。

- `max_iter` (int, default=1000): 优化算法的最大迭代次数。

- `tol` (float, default=1e-4): 优化的停止准则的阈值。

- `selection` ({'cyclic', 'random'}, default='cyclic'): 优化过程中是否随机选择坐标。

**<u>其他参数:</u>**

例如 `precompute`, `warm_start`, `positive`, `random_state` 等参数可以根据模型训练时的需要来设置。

- `fit(X, y)`: 拟合模型。

- `predict(X)`: 使用拟合后的模型进行预测。

- `score(X, y)`: 返回模型的决定系数 R^2。

- `path(X, y)`: 计算正则化路径。
