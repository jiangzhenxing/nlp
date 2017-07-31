from sklearn import svm
import numpy as np
from sklearn import feature_extraction  
from sklearn.feature_extraction.text import TfidfTransformer  
from sklearn.feature_extraction.text import CountVectorizer 
import jieba

docs = []

for doc in ['data/好评.csv', 'data/差评.csv']:
    f = open(doc)
    L = []
    f.readline()	# skip title
    for line in f:
        values = line.split(',')
        comment = values[3]
        L.extend(list(jieba.cut(comment.strip())))
    f.close()
    docs.append(' '.join(L))

stop_words = set(open('data/stop_words.txt').read().split())

#该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频  
vectorizer = CountVectorizer(stop_words=stop_words)
word_count = vectorizer.fit_transform(docs)

#该类会统计每个词语的tf-idf权值
transformer = TfidfTransformer()
tfidfs = transformer.fit_transform(word_count)

# 情感分析使用的关键词
keywords = set()
word = vectorizer.get_feature_names() #获取词袋模型中的所有词语

for f in tfidfs:
    words = list(zip(f.data,f.indices))
    words.sort(key=lambda e: e[0], reverse=True)
    keywords.update([word[w[1]] for w in words[:100]])

print('使用的词有',len(keywords),'个：')
print(keywords)

def feature_labels(f):
    features = []
    labels = []

    for line in open(f):
        comment,score,label = line.strip().split(',')
        labels.append(1 if label=='1' else -1)
        comment = set(comment.split())
        feature = [1 if w in comment else 0 for w in keywords]
        features.append(feature)
    return features,labels

clf = svm.SVC()
train_data, train_y = feature_labels('data/train_data.txt')
clf.fit(train_data, train_y)

test_data, test_y = feature_labels('data/test_data.txt')

predicts = clf.predict(test_data)
true = list(np.array(test_y) == np.array(predicts)).count(True)

print('准确率为：')
print(true/len(test_y))
