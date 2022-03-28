import numpy as np
from .neighbors import OneNN_Scikit

def chattCorr(Z, Y):
    # Y ind Z
    if len(Z.shape)==2:
        if Z.shape[1] ==1:
            Z = np.squeeze(Z)
        else:
            print(Z.shape)
            error("Cannot handle multidimensional Z.")

    if len(Y.shape)==2:
        if Y.shape[1] ==1:
            Y = np.squeeze(Y)
        else:
            print(Y.shape)
            error("Cannot handle multidimensional Y.")

    n = len(Z)
    W = Z

    idx = np.argsort(W)

    sortedX = W[idx]
    sortedY = Y[idx]

    p = np.argsort(sortedY)
    R = np.arange(n)
    tmpR = np.arange(n)
    R[p] = tmpR + 1
    L = (n+1) - R.copy()

    Rshift = R.copy()
    R = np.delete(R, 0)
    Rshift = np.delete(Rshift, -1)

    Tn_num = n*sum(np.abs(R - Rshift))
    Tn_den = 2*np.dot(L, n-L)

    #Tn_num_noties = 3*sum(np.abs(R-Rshift))
    #Tn_den_noties = n**2 -1

    return 1 - Tn_num/Tn_den
    #return 1 - Tn_num_noties/Tn_den_noties


def codec2(Z, Y, n_jobs=None):
	# Y ind Z
    if len(Z.shape)==1:
        Z = Z.reshape(-1,1)
    if len(Y.shape)==2:
    	if Y.shape[1] ==1:
    		Y = np.squeeze(Y)
    	else:
    		print(Y.shape)
    		error("Cannot handle multidimensional Y.")

    n, q = Z.shape
    W = Z
    M = OneNN_Scikit(W)

    p = np.argsort(Y) # ascending
    R = np.arange(n)
    tmpR = np.arange(n)
    R[p] = tmpR + 1

    RM = R[M]
    minRM = n*np.minimum(R, RM)

    L = (n+1) - R

    Tn_num = sum(minRM - L**2)
    Tn_den = np.dot(L, n-L)

    return Tn_num/Tn_den

def codec3(Z, Y, X, n_jobs=None):
    # Y ind Z given X
    
    if len(Z.shape)==1:
        Z = Z.reshape(-1,1)
    if len(X.shape)==1:
        X = X.reshape(-1,1)
    if len(Y.shape)==2:
    	if Y.shape[1] ==1:
    		Y = np.squeeze(Y)
    	else:
    		print(Y.shape)
    		error("Cannot handle multidimensional Y.")

    n, px = X.shape
    N = OneNN_Scikit(X)

    W = np.hstack((X, Z))
    M = OneNN_Scikit(W)

    p = np.argsort(Y) # ascending
    R = np.arange(n)
    tmpR = np.arange(n)
    R[p] = tmpR + 1

    RM = R[M]
    RN = R[N]
    minRM = np.minimum(R,RM)
    minRN = np.minimum(R,RN)    

    Tn_num = sum(minRM - minRN)
    Tn_den = sum(R - minRN)

    return Tn_num/Tn_den 


def main():

    np.random.seed(123436)
    print("Hello CODEC")
    n = 10000
    p = 2
    X = np.random.rand(n,p)
    #X[:,0] = np.arange(n)/n
    #Y = X[:,0]
    #X = np.random.rand(n,p)
    #Y = np.sum(X**2,1)
    Y = np.mod((X[:,0] + X[:,1]), 1.0)
    print('Continuous:')
    print('\tCODEC2 X1 to Y: ', codec2(X[:,1], Y))
    print('\tCODEC2 X1 and X2 to Y: ', codec2(X, Y))
    print('\tCODEC3 X1 to Y Given X2: ', codec3(X[:,0], Y, X[:,1]))

    X = np.random.binomial(size=n, n=1, p=0.5)
    Z = np.random.binomial(size=n, n=1, p=0.5)
    Y = X
    print('Binary:')
    print('\tCODEC3 X1 to Y Given X2: ', codec3(Z,Y,X))
    Xn = X + 0.01*np.random.normal(size=(n))
    Yn = Y + 0.01*np.random.normal(size=(n))
    Zn = Z + 0.01*np.random.normal(size=(n))
    print('Randomized:')
    print('\tCODEC3 X1 to Y Given X2: ', codec3(Zn,Yn,Xn))



if __name__ == '__main__':
	main()
