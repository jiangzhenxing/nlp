使用wiki百科中⽂⽂库的⽂章做为训练语料库，⼤体过程如下：
1. 下载wiki百科的中⽂⽂集(我是用百度⽹盘下载，然后再下载到本地：http://pan.baidu.com/s/1qXOsAs8)
2. 从⽂件中提取wiki⽂本。python3 process_wiki.py zhwiki-latest-pagesarticles.xml.bz2 wiki.zh.text
3. 把⽂章中的繁体字转为简体(使用opencc)。opencc -i wiki.zh.text -o wiki.zh.text.jian -c t2s.json
4. 分词(使用jieba分词软件)。python3 ws.py wiki.zh.text.jian wiki.zh.text.jian.seg
5. 训练word2vec模型(使用gensim)。python train_word2vec_model.py wiki.zh.text.jian.seg wiki.zh.text.model wiki.zh.text.vector
6. 使用word2vec模型计算相近词，如下：
model = gensim.models.Word2Vec.load(“wiki.zh.text.model")
model.most_similar("⾜球")