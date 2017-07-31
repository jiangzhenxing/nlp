import os
import numpy as np
import jieba
from sklearn import feature_extraction  
from sklearn.feature_extraction.text import TfidfTransformer  
from sklearn.feature_extraction.text import CountVectorizer 

from gensim import models,corpora  

docs = []
titles = []
jieba.load_userdict('userdict.txt')

for doc in ['01.txt', '02.txt', '03.txt']:
    f = open('docs/'+doc, encoding='GBK')
    L = []
#    title = f.readline()
#    titles.append(list(jieba.cut(title.strip())))
    for line in f:
        L.extend(list(jieba.cut(line.strip())))
    f.close()
    docs.append(' '.join(L))

stop_words = set(open('stop_words.txt').read().split())

#该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频  
vectorizer = CountVectorizer(stop_words=stop_words)
word_count = vectorizer.fit_transform(docs)

#该类会统计每个词语的tf-idf权值
transformer = TfidfTransformer()
tfidfs = transformer.fit_transform(word_count)

keywords = []

for f in tfidfs:
    words = list(zip(f.data,f.indices))
    words.sort(key=lambda e: e[0], reverse=True)
    keywords.append(words)
    
word = vectorizer.get_feature_names() #获取词袋模型中的所有词语  

word_tfidf = []

for kws in keywords:
    word_tfidf.append(dict([(word[w[1]], w[0]) for w in kws]))

def score(s, word_score):
    return np.average([word_score.get(w,0) for w in s])

def abstract(doc, word_score, n):
    content = []
    index = 0
    for line in doc:
        scores = [(s, score(jieba.cut(s), word_score), index+i)
                    for i,s in enumerate(line.strip().split('。')) if len(s)>0]
        content.extend(scores)
        index += len(scores)
    content.sort(key=lambda c: c[1], reverse=True)
    abst = content[:n]
    abst.sort(key=lambda c: c[2])
    return [s[0] for s in abst]

for f, word_score in zip(['01.txt', '02.txt', '03.txt'], word_tfidf):
    doc = open('docs/'+f, encoding='GBK').readlines()
    abst = abstract(doc, word_score, n=4)
    for s in abst:
        print(s)
    print('-' * 50)


