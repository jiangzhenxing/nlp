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

# 转换为矩阵
A = np.mat(A)
B = np.mat(B)
pi = np.mat(pi).transpose()

def decoding_forward(os):
    '''
    使用前向算法计算输出状态序列的概率
    os: 状态序列
    '''
    if len(os) == 1:
        ps, pos = (pi, 1)                   # 初始的状态分布和序列概率
        next_ps = A.dot(ps)                 # 下一步状态概率分布
        now_po = B[os[-1]-1].dot(pi)[0,0]   # 当前输出的概率
        return (next_ps, now_po * pos)
    else:
        ps, pos = decoding_forward(os[:-1]) # 当前状态概率分布和序列概率
        next_ps = A.dot(ps)                 # 下一步状态概率分布
        now_po = B[os[-1]-1].dot(ps)[0,0]   # 当前输出的概率
        return (next_ps, now_po * pos)

def decoding_backward(os):
    '''
    使用后向算法计算输出状态序列的概率
    '''
    ps = np.mat([1,1,1]).transpose()    # 在不同状态下当前及以后观测序列出现的概率

    # 从后向前计算不同状态下后续观测序列出现的概率
    for i in range(len(os)-1, -1, -1):
        o = os[i]
        tail_po = A * ps    # 当前各状态下出现后续序列的概率
        now_po = B[o-1]     # 当前各状态发射出当前输出的要率
        ps = np.mat(np.array(now_po.transpose()) * np.array(tail_po))

    return (ps, pi.transpose() * ps)

os = [1,6,3,5,2,7,3,5,2,4]

ps, po = decoding_forward(os)

print(','.join(map(str,os)))
print('出现的概率为：')
print(po)
print('状态的概率分布为：')
print(ps)

ps, pos = decoding_backward(os)
print('decoding_backward计算状态出现的概率为：')
print(pos)
print('状态的概率分布为：')
print(ps)
