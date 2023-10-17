https://thecleverprogrammer.com/2022/02/15/flipkart-reviews-sentiment-analysis-using-python/

1. **数据清洗**:

```Python
import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
# 自然语言处理库，用于情感分析。
from nltk.sentiment.vader import SentimentIntensityAnalyzer
# 生成文本数据的词云。
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerato
# 英文词干提取器
stemmer = nltk.SnowballStemmer("english")
# # 设置英文停用词
stopword=set(stopwords.words('english'))

data = pd.read_csv("https://raw.githubusercontent.com/amankharwal/Website-data/master/flipkart_reviews.csv")

def clean(text):
    text = str(text).lower()  # 把文本转换为小写
    # 以下几行使用正则表达式去除URL、HTML标签、标点符号、数字等不需要的内容
    text = re.sub('$.*?$', '', text)  # 删除方括号内的内容
    text = re.sub('https?://\S+|www\.\S+', '', text)  # 删除网址
    text = re.sub('<.*?>+', '', text)  # 删除HTML标签
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)  # 删除标点
    text = re.sub('\n', '', text)  # 删除换行符
    text = re.sub('\w*\d\w*', '', text)  # 删除数字
    # 移除停用词并进行词干提取
    text = [word for word in text.split(' ') if word not in stopword]  # 删除停用词
    text=" ".join(text)  # 合并文本
    text = [stemmer.stem(word) for word in text.split(' ')]  # 词干提取
    text=" ".join(text)  # 合并文本
    return text
data["Review"] = data["Review"].apply(clean)
```

2. **数据可视化**: 使用词云，可以直观地看到用户最常用来描述产品的词汇。

```Python
from wordcloud import WordCloud, STOPWORDS
text = " ".join(i for i in data.Review)
stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)
plt.figure(figsize=(15,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
```

3. **情感分析**: 使用VADER情感分析工具，为每个评论计算正面、负面和中性的分数。

```Python
from nltk.sentiment.vader import SentimentIntensityAnalyzer
# 下载VADER情感分析器的词汇表
nltk.download('vader_lexicon')
# 创建情感分析器对象
sentiments = SentimentIntensityAnalyzer()
# 获取每个评论的情感分数并将其添加到DataFrame
data["Positive"] = [sentiments.polarity_scores(i)["pos"] for i in data["Review"]]
data["Negative"] = [sentiments.polarity_scores(i)["neg"] for i in data["Review"]]
data["Neutral"] = [sentiments.polarity_scores(i)["neu"] for i in data["Review"]]
```

4. **结果评估**: 通过对正面、负面和中性分数进行汇总，我们可以得到对Flipkart评论的总体情感评估。

```Python
x = sum(data["Positive"])
y = sum(data["Negative"])
z = sum(data["Neutral"])
def sentiment_score(a, b, c):
    if (a>b) and (a>c):
        return "正面 😊 "
    elif (b>a) and (b>c):
        return "负面 😠 "
    else:
        return "中性 🙂 "
print(sentiment_score(x, y, z))
```

大部分的Flipkart用户评论都是中性的，但也有许多正面的评论。这意味着Flipkart的产品质量和客户服务仍然受到客户的欢迎。



VADER（Valence Aware Dictionary and sEntiment Reasoner）情感分析工具是一种流行的词典和基于规则的模型

**<u>计算逻辑</u>**

1. **词典**：VADER 有一个单词表（字典），每个单词都被赋予极性得分。例如，“爱”这个单词可能会有积极得分，而“恨”则会有消极得分。
2. **修饰符**：VADER 考虑可以改变短语情感的修饰符。例如，在“非常好”的“非常”一词相比仅使用“好”时增加了情感上的积极性。
3. **否定**：VADER 还考虑否定因素。例如，“不好”的反转了“好”的情绪。
4. **助推器单词**：可以增加（提高）情绪值的单词，如在 “incredibly awesome" 中使用 "incredibly"。
5. **表情符号、俚语和缩略语**: VADER 的字典包括表情符号、俚语和缩略语 (如 "lol" 或 ":)") 的评分，以捕捉来自这些术语的情感，这些术语在社交媒体内容中很常见。
6. **情感得分计算**：
   - 将文本分解为单词或短语。
   - 为每个单词或短语分配一个情感分数（根据词典）。
   - 结合这些分数来得到一个总体的情感分数。
7. **规则增强**：在计算情感分数时，VADER还考虑了各种规则。例如，对于否定词、强调词等的处理。

<u>**分数计算**</u>

当 VADER 分析一段文本时，它会产生四个得分：

- **积极 (pos)**：落入积极类别的文本比例。

- **中性 (neu)**：是中性的文本比例。

- **消极 (neg)**：落入消极类别的文本比例。

- **复合**: 这是一个复合得分，它计算所有词典评级之和，并将其归一化为 -1（非常负面）到 +1（非常正面）之间。如果您想要给定句子的单一单向情感度量，则此指标最有用。

使用以下公式计算复合得分：

$ \text{compound} = \frac{\sum_{i=1}^{N} s_{i}}{\sqrt{(\sum_{i=1}^{N} s_{i}^2) + \alpha}} $

其中:

- $ s_i$ 是第 i 个单词的极性评分。
- N 是句子中单词总数。
- α 是规范化常数（通常由原始作者设置为15以确保复合得分在 -1 和 1 之间）。

根据复合得分，通常做法是将情绪分类为：

* 如果 compound score >0.05，则为积极
* 如果 compound score 在 -0.05 和 0.05 之间，则为中性
* 如果 compound score < -0.05，则为消极

VADER并不完全依赖于统计学方法，而是基于专家制定的规则。它旨在理解互联网语言的细微差别，包括表情符号、俚语和缩略语的使用。它是一种混合模型，结合了词典方法与规则，并且如提供的文本所述，在其预期领域内即使与复杂统计模型相比也具有竞争力。

例如，“Hello, world. I am terrible”这个句子的输出可能是：

- Negativity: 高分（因为“terrible”是负面词）
- Positivity: 低分
- Neutrality: 中等分（因为“Hello, world. I am”是中立的）
- Compound: 一个聚合的分数，表示整个句子的情感。

