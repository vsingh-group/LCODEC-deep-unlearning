import torch
import numpy as np

from .neighbors import OneNN_Torch

def codec2(Z, Y):
	# Y ind Z
    if len(Z.shape)==1:
        Z = Z.reshape(-1,1)
    if len(Y.shape)==2:
    	if Y.shape[1] ==1:
    		Y = Y.squeeze()
    	else:
    		print(Y.shape)
    		error("Cannot handle multidimensional Y.")

    n, q = Z.shape
    W = Z
    M = OneNN_Torch(W)

    p = torch.argsort(Y) # ascending
    R = torch.arange(n)
    tmpR = torch.arange(n)
    R[p] = tmpR + 1

    RM = R[M]
    minRM = n*torch.minimum(R, RM)

    L = (n+1) - R

    Tn_num = (minRM - L**2).sum()
    Tn_den = torch.dot(L, n-L)

    return Tn_num/Tn_den

def codec3(Z, Y, X):
    # Y ind Z given X
    
    if len(Z.shape)==1:
        Z = Z.view(-1,1)
    if len(X.shape)==1:
        X = X.view(-1,1)
    if len(Y.shape)==2:
    	if Y.shape[1] ==1:
    		Y = Y.squeeze()
    	else:
    		print(Y.shape)
    		error("Cannot handle multidimensional Y.")

    n, px = X.shape

    N = OneNN_Torch(X)

    W = torch.hstack((X, Z))
    M = OneNN_Torch(W)

    p = torch.argsort(Y) # ascending
    R = torch.arange(n)
    tmpR = torch.arange(n)
    R[p] = tmpR + 1

    RM = R[M]
    RN = R[N]
    minRM = torch.minimum(R,RM)
    minRN = torch.minimum(R,RN)    

    Tn_num = (minRM - minRN).sum()
    Tn_den = (R - minRN).sum()

    return Tn_num/Tn_den 


def main():

    np.random.seed(123436)
    torch.seed(123456)
    print("Hello Torch CODEC")
    n = 10000
    p = 2
    X = np.random.rand(n,p)
    #X[:,0] = np.arange(n)/n
    #Y = X[:,0]
    #X = np.random.rand(n,p)
    #Y = np.sum(X**2,1)
    Y = np.mod((X[:,0] + X[:,1]), 1.0)
    X = torch.Tensor(X)
    Y = torch.Tensor(Y)
    print('Continuous:')
    print('\tCODEC2 X1 to Y: ', codec2(X[:,1], Y))
    print('\tCODEC2 X1 and X2 to Y: ', codec2(X, Y))
    print('\tCODEC3 X1 to Y Given X2: ', codec3(X[:,0], Y, X[:,1]))

    X = torch.Tensor(np.random.binomial(size=n, n=1, p=0.5))
    Z = torch.Tensor(np.random.binomial(size=n, n=1, p=0.5))
    Y = X
    print('Binary:')
    print('\tCODEC3 X1 to Y Given X2: ', codec3(Z,Y,X))
    Xn = X + torch.Tensor(0.01*np.random.normal(size=(n)))
    Yn = Y + torch.Tensor(0.01*np.random.normal(size=(n)))
    Zn = Z + torch.Tensor(0.01*np.random.normal(size=(n)))
    print('Randomized:')
    print('\tCODEC3 X1 to Y Given X2: ', codec3(Zn,Yn,Xn))



if __name__ == '__main__':
	main()
