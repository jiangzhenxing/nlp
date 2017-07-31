#!/usr/bin/env python3
'''
使用EM算法求解IBM Model 1的翻译概率.
示例来自“Statistical Machine Translation” by Koehn
'''
pairs = [(['the','house'], ['das','haus']),
         (['the','book'], ['das','buch']),
         (['a','book'], ['ein','buch'])]

fwords = ['das','ein','buch','haus']
ewords = ['the','a','book','house']

def init_t():
    '''
    init t(e|f)=1/len(ewords)
    '''
    t = {}
    for f in fwords:
        tf = dict([(e,1/len(ewords)) for e in ewords])
        t[f] = tf
    return t

def init_c():
    '''
    init count(e|f)=0
    '''
    c = {}
    for f in fwords:
        cf = dict([(e,0) for e in ewords])
        c[f] = cf
    return c

def em4model1(pairs, iterations=100):
    '''
    EM training algorithm for IBM Model 1.
    pairs: set of sentence pairs (e,f)
    '''
    t = init_t()  # t(e|f)
    for i in range(iterations):
        # initialize
        c = init_c()      # 0 count(e|f)
        total = dict([(f,0) for f in fwords])  # 0 for all f
        # compute normalization
        for se,sf in pairs:
            s_total = {}
            for e in se:
                s_total[e] = 0
                for f in sf:
                    s_total[e] += t[f][e]
            # collect counts
            for e in se:
                for f in sf:
                    c[f][e] += t[f][e] / s_total[e]
                    total[f] += t[f][e] / s_total[e]
        # estimate probabilities
        for f in fwords:
            for e in ewords:
                t[f][e] = c[f][e] / total[f]

    return t


#print('t', init_t())
#print('c', init_c())
t = em4model1(pairs)
for f,te in t.items():
    print(f, te)

