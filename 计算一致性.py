# 导入必要的库
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from gensim.models import Phrases
from gensim.models.phrases import Phraser
from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel
from gensim.models import CoherenceModel
import numpy as np
from numpy.linalg import norm
from tqdm import tqdm  # 用于显示进度条

# 下载停用词数据集
nltk.download('stopwords')

# 定义停用词集合
stop_words = set(stopwords.words('english'))

# 初始化Porter Stemmer进行词干化
ps = PorterStemmer()

# 文本清洗函数
def clean_text(text):
    # 处理缺失值（NaN）
    if pd.isna(text):
        return ""

    # 1. 转为小写
    text = text.lower()

    # 2. 去除标点符号
    text = re.sub(r'[^\w\s]', '', text)

    # 3. 去除数字
    text = re.sub(r'\d+', '', text)

    # 4. 拆分单词并去除停用词
    words = text.split()
    words = [word for word in words if word not in stop_words]

    # 5. 词干化
    words = [ps.stem(word) for word in words]

    # 6. 重新组合为单个字符串
    return ' '.join(words)

# 读取CSV文件
data = pd.read_csv('Artikelrezensionen_EW_2018_2023_translated.csv', sep='|')

# 示例：添加Review_Type列（根据实际情况进行调整）
# 这里我们假设某些特定的用户是影评人，其他的是普通用户
critic_users = ['Felix Felicis', 'Mrs Snape', 'Draco (nv)']  # 示例名单
data['Review_Type'] = data['User'].apply(lambda x: 'critic' if x in critic_users else 'user')

# 对Content和Comment列进行文本清洗
data['Cleaned_Content'] = data['Content'].apply(clean_text)
data['Cleaned_Comment'] = data['Comment'].apply(clean_text)

# 合并Content和Comment列，形成一个完整的评论文本
data['Combined_Text'] = data['Cleaned_Content'] + " " + data['Cleaned_Comment']

# 创建语料库（每一条评论文本作为语料的一部分）
corpus = data['Combined_Text'].tolist()

# 将每个文本条目转换为单词列表
texts = [text.split() for text in corpus]

# 生成双词短语（Bigram）
bigram = Phrases(texts, min_count=5, threshold=100)
bigram_mod = Phraser(bigram)

# 将每个文档转换为bigram格式
texts_bigram = [bigram_mod[doc] for doc in texts]

# 打印处理后的bigram文本示例
print("Bigram example:", texts_bigram[0])

### 步骤3：LDA模型训练

# 创建词典
id2word = Dictionary(texts_bigram)

# 创建语料库（文档-词袋）
corpus_bow = [id2word.doc2bow(text) for text in texts_bigram]

# 训练LDA模型
num_topics = 25  # 假设最佳主题数为25
lda_model = LdaModel(corpus=corpus_bow,
                     id2word=id2word,
                     num_topics=num_topics,
                     random_state=100,
                     update_every=1,
                     chunksize=100,
                     passes=10,
                     alpha='auto',
                     per_word_topics=True)

# 查看每个主题最相关的双词
for idx, topic in lda_model.print_topics(-1):
    print(f"Topic: {idx} \nWords: {topic}\n")


# 定义加权余弦相似度函数
def weighted_cosine_similarity(vec1, vec2, weights):
    numerator = np.sum(weights * vec1 * vec2)
    denominator = np.sqrt(np.sum(weights * vec1 ** 2)) * np.sqrt(np.sum(weights * vec2 ** 2))
    if denominator == 0:
        return 0
    return numerator / denominator

# 获取每条Content和Comment的主题分布
def get_topic_distribution(text, bigram_mod, id2word, lda_model):
    # 转换为单词列表
    words = text.split()

    # 应用双词模型
    bigram_words = bigram_mod[words]

    # 转换为bag-of-words
    bow = id2word.doc2bow(bigram_words)

    # 获取主题分布
    topic_dist = lda_model.get_document_topics(bow, minimum_probability=0)

    # 转换为向量
    topic_vector = np.array([prob for _, prob in topic_dist])
    return topic_vector

# 计算主题权重（w）
# 这里我们将使用影评人和用户评论的平均主题分布作为权重
# 首先，获取影评人和用户评论的文本
critic_comments = data[data['Review_Type'] == 'critic']['Combined_Text'].tolist()
user_comments = data[data['Review_Type'] == 'user']['Combined_Text'].tolist()

# 获取主题分布
critic_topic_probs = [get_topic_distribution(text, bigram_mod, id2word, lda_model) for text in critic_comments]
user_topic_probs = [get_topic_distribution(text, bigram_mod, id2word, lda_model) for text in user_comments]

# 转换为numpy数组
critic_topic_probs = np.array(critic_topic_probs)
user_topic_probs = np.array(user_topic_probs)

# 计算影评人评论和用户评论的平均主题分布
critic_topic_mean = np.mean(critic_topic_probs, axis=0)
user_topic_mean = np.mean(user_topic_probs, axis=0)

# 计算权重向量 w
w = (critic_topic_mean + user_topic_mean) / 2

# 确保权重向量归一化或适当缩放（根据需要调整）
w_normalized = w / np.linalg.norm(w)

print("主题重要性权重 (w):", w_normalized)

# 初始化列表存储一致性和差异性指标
similarities = []
differences = []

# 使用tqdm显示进度条
for index, row in tqdm(data.iterrows(), total=data.shape[0]):
    content = row['Cleaned_Content']
    comment = row['Cleaned_Comment']

    # 获取主题分布
    content_topic = get_topic_distribution(content, bigram_mod, id2word, lda_model)
    comment_topic = get_topic_distribution(comment, bigram_mod, id2word, lda_model)

    # 计算加权余弦相似度（一致性）
    sim = weighted_cosine_similarity(content_topic, comment_topic, w_normalized)
    similarities.append(sim)

    # 计算加权差异性（用户评论相对于内容的差异）
    diff_vector = comment_topic - content_topic
    # 只保留正差异（即用户评论中主题概率高于内容的部分）
    positive_diff = np.maximum(diff_vector, 0)
    # 加权差异
    weighted_positive_diff = w_normalized * positive_diff
    # 计算差异的L2范数
    diff = np.linalg.norm(weighted_positive_diff)
    differences.append(diff)

# 将一致性和差异性指标添加到数据集中
data['Content_Comment_Consistency'] = similarities
data['Comment_vs_Content_Difference'] = differences

# 查看前5条数据
print(data[['Content', 'Comment', 'Content_Comment_Consistency', 'Comment_vs_Content_Difference']].head())

# 保存结果到新的CSV文件
data.to_csv('Artikelrezensionen_With_Consistency_and_Difference.csv', index=False)