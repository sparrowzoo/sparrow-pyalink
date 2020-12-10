from math import *
import numpy
import matplotlib.pyplot as plt

def matrix_factorization(R,P,Q,K,steps=5000,alpha=0.0002,beta=0.02): #矩阵因子分解函数，steps：梯度下降次数；alpha：步长；beta：β。
    Q=Q.T                 # .T操作表示矩阵的转置
    result=[]
    for step in range(steps): #梯度下降
        for i in range(len(R)):
            for j in range(len(R[i])):
                    eij=R[i][j]-numpy.dot(P[i,:],Q[:,j])       # .DOT表示矩阵相乘
                    for k in range(K):
                      if R[i][j]>0:        #限制评分大于零
                        P[i][k]=P[i][k]+alpha*(2*eij*Q[k][j]-beta*P[i][k])   #增加正则化，并对损失函数求导，然后更新变量P
                        Q[k][j]=Q[k][j]+alpha*(2*eij*P[i][k]-beta*Q[k][j])   #增加正则化，并对损失函数求导，然后更新变量Q
        eR=numpy.dot(P,Q)
        e=0
        for i in range(len(R)):
            for j in range(len(R[i])):
              if R[i][j]>0:
                    e=e+pow(R[i][j]-numpy.dot(P[i,:],Q[:,j]),2)      #损失函数求和
                    for k in range(K):
                        e=e+(beta/2)*(pow(P[i][k],2)+pow(Q[k][j],2)) #加入正则化后的损失函数求和
        result.append(e)
        if e<0.001:           #判断是否收敛，0.001为阈值
            break
    return P,Q.T,result

if __name__ == '__main__':   #主函数
    R=[                 #原始矩阵
        [1,1,2,3],
        [1,1,1,1],
        [2,1,1,2],
        [3,1,2,1]
    ]
    R=numpy.array(R)
    N=len(R)    #原矩阵R的行数
    M=len(R[0]) #原矩阵R的列数
    K=2   #K值可根据需求改变
    P=numpy.random.rand(N,K) #随机生成一个 N行 K列的矩阵
    Q=numpy.random.rand(M,K) #随机生成一个 M行 K列的矩阵
    nP,nQ,result=matrix_factorization(R,P,Q,K)
    print(R)         #输出原矩阵
    print(nP)
    print(nQ)
    R_MF=numpy.dot(nP,nP.T)



    print(R_MF)      #输出新矩阵
    #画图
    plt.plot(range(len(result)),result)
    plt.xlabel("time")
    plt.ylabel("loss")
    plt.show()
