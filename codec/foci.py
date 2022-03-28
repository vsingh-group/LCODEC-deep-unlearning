import numpy as np
from .codec import codec2, codec3


# feature ordering
def foci(X, Y, earlyStop=True, verbose=False, n_jobs=None):
    p = X.shape[1]

    indeps = np.empty((p,1))
    maxval = -100
    maxind = None
    for i in range(p):
        tmp = codec2(X[:,i], Y, n_jobs=n_jobs)
        if tmp > maxval:
            maxval = tmp
            maxind = i

    assert(maxval > -100)
    all_inds = np.arange(p)

    deplist = [maxind]
    depset = set(deplist)
    
    indepset = set(all_inds).difference(depset)
    indeplist = list(indepset)

    ordering = [maxind]
    codecVals = [maxval]

    for k in range(p-1):
        assert(list(depset.intersection(indepset))==[])
        assert(len(list(depset.union(indepset)))==p)

        if verbose:
            print(maxval)
            print(deplist)
            print(indeplist)
        cX = X[:,deplist]

        condeps = np.empty((len(indeplist),1))
        maxval = -100
        mostdepL = None
        for l in indeplist:
            cZ = X[:,l]
            tmp = codec3(cZ, Y, cX, n_jobs=n_jobs)

            if tmp > maxval:
                maxval = tmp
                mostdepL = l

        # pick randomly (the last one) if all -inf
        if maxval <= -100:
            mostdepL = l

        if maxval <= 0.0 and earlyStop:
            break
            
        depset.add(mostdepL)
        indepset.remove(mostdepL)

        deplist.append(mostdepL)
        indeplist = list(indepset)

        ordering.append(mostdepL)
        codecVals.append(maxval)

    return ordering, codecVals

# feature ordering
# identifies the top, most dependent feature with Y
def cheap_foci(X, Y, n_jobs=None):
    p = X.shape[1]

    indeps = np.empty((p,1))
    maxval = -100
    maxind = None
    for i in range(p):
        tmp = codec2(X[:,i], Y, n_jobs=n_jobs)
        if tmp > maxval:
            maxval = tmp
            maxind = i

    ordering = [maxind]
    codecVals = [maxval]

    return ordering, codecVals


def createFOCIGraph(X):
    p = X.shape[1]
    graph = np.eye(p)

    for i in range(p):
        other_inds = list(np.arange(p))
        other_inds.remove(i)
        
        xs, vals = foci(X[:,other_inds], X[:,i], earlyStop=True, verbose=False)

        for k in range(len(xs)):
            x = xs[k]
            val = vals[k]
            idx = other_inds[x]
            print('\t',i, k, x, idx, val)

            graph[i, idx] = val

    return graph


def main():

    ## Compare Time for multijobs
    print('#### Time Comparion for MultiJob Scikit NN ####')
    n = 1000
    p = 1000
    X = np.random.rand(n,p)
    Y = np.random.rand(n,1)

    import time
    tic = time.time()
    tmp = foci(X, Y, n_jobs=None, verbose=False)
    myt = time.time() - tic
    print(f'Time for n_jobs=1: {myt} seconds.')
    
    tic = time.time()
    tmp = foci(X, Y, n_jobs=-1, verbose=False)
    myt = time.time() - tic
    print(f'Time for n_jobs=-1: {myt} seconds.')


if __name__ == '__main__':
    main()



