import numpy as np

'''
A crucial question is the following: 
given a sentence s, how do we find the highest scoring parse tree for s, or more explicitly, 
how do we find
arg max p(t) t∈T (s) ?
This section describes a dynamic programming algorithm, the CKY algorithm, for this problem.
From "Probabilistic Context-Free Grammars (PCFGs)" by Michael Collins
'''
S = 'the man saw the dog with the telescope'.split()
#S = 'the man saw the dog'.split()

q = {'S'  : {('NP','VP') : 1.0},
     'VP' : {('Vt','NP') : 0.8,
     	     ('VP','PP') : 0.2},
     'NP' : {('DT','NN') : 0.8,
     	     ('NP','PP') : 0.2},
     'PP' : {('IN','NP') : 1.0},
     'Vi' : {'sleeps': 1.0},
     'Vt' : {'saw': 1.0},
     'NN' : {'man': 0.1,
     	     'woman': 0.1,
     	     'telescope': 0.3,
     	     'dog': 0.5},
     'DT' : {'the': 1.0},
     'IN' : {'with': 0.6,
             'in': 0.4}
    }


def pi(i,j,X):
    #print('pi', i, j, X)
    if i == j:
        #print('终止：', i, j, X, S[i], q[X])
        return S[i], q[X][(S[i])] if (S[i]) in q[X] else 0
    
    maxp = 0
    maxt = ()
    subt = q[X]
    #print('sub tree:', subt, i, j)
    for s in range(i,j,1):
        for t,qt in subt.items():
            #print(t, qt)
            if type(t) == str:
                #print('不可能：',i,j,X,(),0)
                return (i,j,X), 0
            sub_max_t1, sub_max_p1 = pi(i,s,t[0])
            #print('sub max tree1:', i,s,t[0], sub_max_t1, 'p:', sub_max_p1)
            sub_max_t2, sub_max_p2 = pi(s+1,j,t[1])
            #print('sub max tree2:', s+1, j, t[1], sub_max_t2, 'p:', sub_max_p2)
            p = qt * sub_max_p1 * sub_max_p2
            #print(i,s,j,X,'p is:', p, '*' * 20)

            #if i == 2 and j == 4 and p > 0:
            #    print(i,s,p,qt,sub_max_p1,sub_max_p2,'*'*50)
            if p > maxp:
                maxp = p
                maxt = ((t[0],sub_max_t1), (t[1],sub_max_t2))
    #print('return', i,j,X,maxt,maxp)
    return maxt, maxp

result = pi(0, len(S)-1, 'S')
print('-------------- result -----------------')
print('S:', result)
