import jieba

goodf = open('data/好评.csv')   # 1005条记录
badf = open('data/差评.csv')    # 1009条记录

# skip title
goodf.readline()
badf.readline()

goods = []
bads = []

def parseComment(c):
    values = c.split(',')
    if len(values) < 9:
        print(c)
        return ('','0')
    comment = values[3]
    seg = jieba.cut(comment)
    score = values[5][-1]
    return ' '.join(seg), score

# num,area,com_client,comment,goods_name,score,times,user_grade,user_id
goods = [ parseComment(line) for line in goodf]
bads = [ parseComment(line) for line in badf]

# comment,score,label
train_data = open('data/train_data.txt', 'w')
test_data = open('data/test_data.txt', 'w')

for d in goods[:800]:
    train_data.write(d[0] + ',' + d[1] + ',' + '1\n')
for d in bads[:800]:
    train_data.write(d[0] + ',' + d[1] + ',' + '0\n')

for d in goods[800:]:
    test_data.write(d[0] + ',' + d[1] + ',' + '1\n')
for d in bads[800:]:
    test_data.write(d[0] + ',' + d[1] + ',' + '0\n')

train_data.close()
test_data.close()

