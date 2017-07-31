import sys
import jieba

inf, outf = sys.argv[1:]
print(inf, outf)

out = open(outf, 'w')

for line in open(inf):
    seg_list = jieba.cut(line.strip())  # 默认是精确模式
    out.write(' '.join(seg_list) + '\n')

out.close()


