import numpy as np

# 转移矩阵
A = [[1/3, 1/3, 1/3],
     [1/3, 1/3, 1/3],
     [1/3, 1/3, 1/3]]

# 发射矩阵
B = [[1/4, 1/6, 1/8],
     [1/4, 1/6, 1/8],
     [1/4, 1/6, 1/8],
     [1/4, 1/6, 1/8],
     [0,   1/6, 1/8],
     [0,   1/6, 1/8],
     [0,   0,   1/8],
     [0,   0,   1/8]]

# 初始状态概率分布
pi = [1/3, 1/3, 1/3]

# 转换为np.array
A = np.array(A)
B = np.array(B)
pi = np.array(pi)

def max(s):
    '''
    求一个序列的最大值并给出其索引
    '''
    index = 0   # 最大值索引
    m = 0       # 最大值
    for i,e in enumerate(s):
        if e > m:
            m = e
            index = i
    return (index, m)

def viterbi(os):
    '''
    使用viterbi算法计算输出序列概率最大的状态序列
    os: 输出序列
    '''
    o = os[0]
    ps = pi * B[o-1]  # 不同状态发射第一个观测序列的概率
    #print('ps: ', ps)
    ss = []                  # 最大概率的状态序列

    for o in os[1:]:
        s = []
        ps2 = []
        # 求不同状态求转移到的最大概率
        for i in range(len(ps)):
            index, m = max(ps * A[i]) # 转移到状态i的最大概率和对应状态
            # print(i, index, m)
            ps2.append(m)
            s.append(index)
        ss.append(s)
        # 不同状态发射当前序列的概率
        ps = ps2 * B[o-1]
    
    # 至最后观测序列概率最大的状态及其概率
    i,p = max(ps)
    #print(ss)

    # 求出概率最大的状态序列
    seq = [i]
    ss.reverse()
    for s in ss:
        seq.insert(0, s[i])
        i = s[i]
    
    return (seq, p)

sname = ['D4', 'D6', 'D8']
os = [1,6,3,5,2,7,3,5,2,4]
#os = [1,6,7]
seq, p = viterbi(os)

print('观测序列为：')
print(os)
print('概率最大的状态为：')
print([sname[s] for s in seq])
print('概率为：')
print(p)
