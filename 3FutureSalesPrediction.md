https://thecleverprogrammer.com/2022/03/01/future-sales-prediction-with-machine-learning/

**<u>数据集概览</u>**

数据集涵盖了在多个广告平台上的广告投入成本及其对应的产品销售数量。主要包含以下四个字段：
1. **TV:** 在电视上的广告花费（美元）；
2. **Radio:** 在广播上的广告花费（美元）；
3. **Newspaper:** 在报纸上的广告花费（美元）；
4. **Sales:** 销售数量（单位）。

在这个环境下，我们的目标变量（依赖变量）是“Sales”，而独立变量则包括TV、Radio和Newspaper的广告花费。

**<u>线性模型模型</u>**

线性回归（Linear Regression）是监督学习的一种，它试图通过建立变量间线性关系的模型来解释变量之间的关联。在简单线性回归中，我们假定输出与输入之间存在线性关系。在多元线性回归中，我们将考虑多个输入特征对输出的影响。线性回归模型可以表示为：

$ \text{Sales} = \beta_0 + \beta_1 \times \text{TV} + \beta_2 \times \text{Radio} + \beta_3 \times \text{Newspaper} + \epsilon $

其中，$\beta_0, \beta_1, \beta_2,$ 和 $\beta_3$ 是模型的参数，$\epsilon$ 是误差项。

通过散点图和相关性分析，我们可以初步了解不同广告支出与销售额之间的关系。这一步能帮助我们判断哪一种广告渠道可能对销售额的影响最大

```python
import plotly.express as px
import plotly.graph_objects as go
figure = px.scatter(data_frame = data, x="Sales",
                    y="TV", size="TV", trendline="ols")
figure.show()

figure = px.scatter(data_frame = data, x="Sales",
                    y="Newspaper", size="Newspaper", trendline="ols")
figure.show()

figure = px.scatter(data_frame = data, x="Sales",
                    y="Radio", size="Radio", trendline="ols")
figure.show()

correlation = data.corr()
print(correlation["Sales"].sort_values(ascending=False))
```

利用`sklearn`中的`LinearRegression`模型，我们能够很直观地进行模型训练和预测。

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

x = np.array(data.drop(["Sales"], 1))
y = np.array(data["Sales"])
xtrain, xtest, ytrain, ytest = train_test_split(x, y, 
                                                test_size=0.2, 
                                                random_state=42)

model = LinearRegression()
model.fit(xtrain, ytrain)
#我们使用R方来评估模型的性能
print(model.score(xtest, ytest))

#features = [[TV, Radio, Newspaper]]
features = np.array([[230.1, 37.8, 69.2]])
print(model.predict(features))
```

在模型训练过程中，算法会自动寻找最佳的参数（$\beta_0, \beta_1, \beta_2, \beta_3$）来最小化真实值和预测值之间的误差。



**<u>普通最小二乘法 (OLS)</u>**

**使用情境**：数据特征无明显多重共线性，模型不易过拟合。

在OLS中，目标函数是最小化残差平方和 (RSS)，

$ J(\beta) = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 $
$ = \sum_{i=1}^{n} (y_i - (\beta_0 + \beta_1 x_{i1} + \beta_2 x_{i2} + \ldots + \beta_p x_{ip}))^2 $

其中:

- $y_i$ 是观测值，
- $\hat{y}_i$ 是预测值，
- $\beta = (\beta_0, \beta_1, \beta_2, \ldots, \beta_p)$ 是模型参数，
- $n$ 是样本数量，
- $p$ 是特征数量。

<u>什么是正规方程 normal equation</u>

正规方程是一个用于寻找使得目标函数最小的参数的解析方法。如你所示的公式，我们可以通过下面的正规方程求得权重 $\boldsymbol{\beta}$：

$ \boldsymbol{\beta} = (X^T X)^{-1} X^T \textbf{y} $

其中，$X$ 是设计矩阵，其每一列对应一个特征，每一行对应一个样本。

假设我们有以下简单线性模型：

$ \hat{y} = \beta_0 + \beta_1 x_1 $

目标函数可以被展开为：

$ J(\beta) = \sum_{i=1}^{n} (y_i - \beta_0 - \beta_1 x_{i1})^2 $

要找到最小化目标函数的 $\beta$ 值，我们对 $\beta_0$ 和 $\beta_1$ 求偏导数，并设为0：

$ \frac{\partial J}{\partial \beta_0} = -2\sum_{i=1}^{n}(y_i - \beta_0 - \beta_1 x_{i1}) = 0 $
$ \frac{\partial J}{\partial \beta_1} = -2\sum_{i=1}^{n}x_{i1}(y_i - \beta_0 - \beta_1 x_{i1}) = 0 $

这样我们得到了两个方程，通过解这两个方程，我们可以得到 $\beta_0$ 和 $\beta_1$ 的值。

正规方程和目标函数最小化之间的关系紧密相连。在线性回归问题中，我们的目标是找到一组参数 $\boldsymbol{\beta}$，使得模型预测的输出 $\hat{y}$ 与实际的输出 $y$ 之间的差的平方和最小——这就是要最小化的目标函数 $J(\beta)$。

>**目标函数最小化**：目标函数 $J(\beta)$ 是模型预测误差的平方和，我们通过最小化目标函数来找到最优参数 $\boldsymbol{\beta}$。在这个过程中，我们将目标函数 $J(\beta)$ 对每个参数 $\beta_j$ 求偏导数，然后将偏导数设为0。
>
>**正规方程**：正规方程提供了一个直接找到最小化目标函数的参数 $\boldsymbol{\beta}$ 的解析解。不需要进行迭代优化，直接通过一个方程就能找到使 $J(\beta)$ 最小的 $\boldsymbol{\beta}$ 值。正规方程是通过将目标函数对参数的梯度（偏导数）设为零而得到的。

在数学层面上，两者的关系是：

我们想要最小化目标函数，即找到 $\frac{\partial J(\beta)}{\partial \beta_j} = 0$ 的 $\boldsymbol{\beta}$。通过对 $J(\beta)$ 求导并设等于0，我们得到了正规方程。正规方程实际上就是目标函数通过梯度为0得到的解析解。

$ \frac{\partial J(\beta)}{\partial \boldsymbol{\beta}} = 0 \Rightarrow \boldsymbol{\beta} = (X^T X)^{-1} X^T \textbf{y} $

单变量线性回归的求解过程是如何一步步发展到多变量线性回归，并采用正规方程法解决的?

> 首先，我们有目标函数（损失函数）：
> $ J(\beta) = \sum_{i=1}^{n} (y_i - \beta_0 - \beta_1 x_{i1})^2 $
>
> 在单变量线性回归中，我们通过对参数 $\beta_0$ 和 $\beta_1$ 分别求偏导数，并令其等于0来找到损失函数最小的点。即：
> $ \frac{\partial J}{\partial \beta_0} = -2\sum_{i=1}^{n}(y_i - \beta_0 - \beta_1 x_{i1}) = 0 $
> $ \frac{\partial J}{\partial \beta_1} = -2\sum_{i=1}^{n}x_{i1}(y_i - \beta_0 - \beta_1 x_{i1}) = 0 $
>
> 这两个方程形成了一个线性方程组，我们可以求解该方程组得到 $\beta_0$ 和 $\beta_1$ 的值。
>
> 而在多元线性回归中，我们有更多的自变量，于是目标函数变为：
> $ J(\beta) = \sum_{i=1}^{n} (y_i - (\beta_0 + \beta_1 x_{i1} + \beta_2 x_{i2} + \ldots + \beta_p x_{ip}))^2 $
> 其中 p 是特征（自变量）数量。
>
> 我们希望通过求解如下方程找到损失函数最小的点：
> $ \frac{\partial J(\beta)}{\partial \boldsymbol{\beta}} = 0 $
>
> 此处 $\boldsymbol{\beta}$ 为参数向量。要找到 $\boldsymbol{\beta}$，我们将问题转化为矩阵形式，目标函数用矩阵表示为：
> $ J(\beta) = (\textbf{y} - X\boldsymbol{\beta})^T (\textbf{y} - X\boldsymbol{\beta}) $
> 其中，
> - $\textbf{y}$ 是一个 n 维输出向量，
> - $X$ 是一个 n×p 的输入矩阵（设计矩阵），
> - $\boldsymbol{\beta}$ 是 p 维参数向量。
>
> 我们对 $\boldsymbol{\beta}$ 求偏导：
> $ \frac{\partial J(\beta)}{\partial \boldsymbol{\beta}} = -2X^T(\textbf{y} - X\boldsymbol{\beta}) $
> 令其等于 0 可得
> $ X^T(\textbf{y} - X\boldsymbol{\beta}) = 0 $
> $ X^T\textbf{y} - X^TX\boldsymbol{\beta} = 0 $
> 解得正规方程的解：
> $ \boldsymbol{\beta} = (X^T X)^{-1} X^T \textbf{y} $
>
> 正规方程法给出了参数 $\boldsymbol{\beta}$ 的解析解。在多元线性回归的情境下，不论特征数量，我们可以直接得到参数的精确解。当我们处于单变量线性回归的情境（只有 $\beta_0$ 和 $\beta_1$）时，这个解析解与通过求解偏导数方程组得到的解是一样的。

**<u>岭回归 (Ridge Regression)</u>**

岭回归通过在OLS的基础上增加L2正则项来处理共线性和过拟合，最小化目标函数：

$ J(\beta) = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^{p} \beta_j^2 $

其中，第二项是L2正则化项，$\lambda$ 是正则化参数。当 $\lambda$ 增加，正则化的强度增加。求解 $\beta$ 可用以下公式：

$ \beta = (X^T X + \lambda I)^{-1} X^T y $

**情景辅助**：例如，在预测房价时，如果使用多个高度相关的特征（如卫生间数量和卧室数量），可使用岭回归防止模型过拟合，并稳定系数估计。

在OLS的基础上增加了一个L2正则项（系数的平方和）。这帮助它处理多重共线性问题。

```python
from sklearn.linear_model import Ridge
model_ridge = Ridge(alpha=1.0).fit(X_train, y_train)  # alpha 是正则化强度
```

**<u>套索回归 (Lasso Regression)</u>**

Lasso回归使用L1正则项，可以产生稀疏解（某些参数为0），通常用于特征选择：

$ J(\beta) = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 + \lambda \sum_{j=1}^{p} |\beta_j| $

第二项是L1正则化项。

Lasso的求解涉及到“软阈值”运算，Lasso回归的解不像岭回归那样有闭式解，通常使用坐标下降法或梯度下降法来求解。

**使用情境**：当模型包含许多不必要的特征时，产生稀疏模型。在包含大量特征但部分特征可能无关紧要的基因表达数据分析中，使用Lasso能自动进行特征选择，只保留一部分重要的特征。相较于OLS，Lasso使用L1正则项（系数的绝对值之和），能够产生稀疏模型（某些系数为0）。

```python
from sklearn.linear_model import Lasso
model_lasso = Lasso(alpha=1.0).fit(X_train, y_train)  # alpha 是正则化强度
```

**<u>弹性网络 (Elastic Net)</u>**

弹性网络结合了L1和L2正则化，它包含两个正则项：

$ J(\beta) = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 + \lambda_1 \sum_{j=1}^{p} |\beta_j| + \lambda_2 \sum_{j=1}^{p} \beta_j^2 $

与Lasso相似，弹性网络通常也使用数值优化方法求解。

**情景辅助**：在金融风险预测中，往往包含大量的宏观经济指标特征，弹性网络可以在保留重要特征的同时，避免过拟合和解决特征间的多重共线性问题。

```python
from sklearn.linear_model import ElasticNet
model_enet = ElasticNet(alpha=1.0, l1_ratio=0.5).fit(X_train, y_train)  # alpha 是正则化强度，l1_ratio 控制两种正则项的比例
```



总结
- **OLS**：若样本量充足且特征间无严重共线性，且关心参数的解释，可使用。
- **Ridge**：若数据存在多重共线性（特征间高度相关），使用Ridge能够稳定参数估计。
- **Lasso**：若希望模型自动进行特征选择，压缩无用特征的系数至0，选用Lasso。
- **Elastic Net**：当既想保留Ridge的稳定性，又想Lasso能做特征选择时，可用弹性网络。

- 所有方法都是通过学习最佳的系数来构建一个线性预测模型。
- 除OLS外，其他方法通过加入不同的正则化项来避免过拟合和处理共线性问题。
- 正则化强度和正则项的选择（L1或L2或两者）影响模型的系数，可能产生不同的模型解释和预测性能。
  

这些方法（最小二乘法、岭回归、Lasso回归和弹性网络）都是线性回归的变种或扩展，它们共享一个核心思想，即通过线性组合输入特征来预测输出变量。其主要的区别在于它们如何估计模型参数（即系数）。例如OLS适用于基础线性建模，岭回归适用于处理多重共线性问题，Lasso和弹性网络在处理具有许多不重要特征的数据集时特别有用。在实际应用中，选择哪个模型通常取决于数据特性和问题背景。例如，在p（特征数）>n（样本数）或存在多重共线性时，岭回归和Lasso可能比OLS更优。而在特征选择方面，Lasso和弹性网络可能比岭回归更有优势。



