import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors

def OneNN_Torch(X, p=2):
    '''
        Compute pairwise p-norm distance and gets
        elements with closest distance.
    '''

    # number of samples is first dimension
    # feature space size is second dimension
    n, d = X.shape

    pdists = torch.cdist(X, X, p=p,
                    compute_mode='use_mm_for_euclid_dist_if_necessary')

    pdists.fill_diagonal_(float('inf'))
    oneNN = torch.argmin(pdists, dim=1)

    return oneNN


def OneNN_Scikit(X, p=2):

    nbrs = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(X)
    distances, indices = nbrs.kneighbors(X)

    return indices[:,1]



def main():

    np.random.seed(12345)
    import time

    print('Correctness:')
    n = 10000
    p = 1
    X = np.random.rand(n, p).astype(np.float16)

    sci = OneNN_Scikit(X)
    print('\tScikit:\t\t', sci[:10])
    tor = OneNN_Torch(torch.Tensor(X)).cpu().numpy()
    print('\tTorch CPU:\t', tor[:10])

    import pdb; pdb.set_trace()

    if torch.cuda.is_available():
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        Xcuda = torch.Tensor(X).to(device)
        gpu = OneNN_Torch(Xcuda).cpu().numpy()[:10]
        print('\tTorch CUDA/GPU:\t', gpu)

    ps = [10, 100, 1000, 10000, 100000, 100000]
    n = 1000

    for p in ps:
        print(f'n = {n}, p = {p}')

        X = np.random.rand(n, p)

        tic = time.time()
        sci = OneNN_Scikit(X)
        toc = time.time()
        print(f'\t Sklearn:\t {toc-tic:.5} seconds.')

        Xcpu = torch.Tensor(X).to('cpu')

        tic = time.time()
        cpu = OneNN_Torch(Xcpu)
        toc = time.time()
        print(f'\t Torch CPU:\t {toc-tic:.5} seconds.')

        if torch.cuda.is_available():
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            Xcuda = torch.Tensor(X).to(device)

            tic = time.time()
            torch.cuda.synchronize()
            gpu = OneNN_Torch(Xcuda)
            torch.cuda.synchronize()
            toc = time.time()
            print(f'\t Torch CUDA/GPU: {toc-tic:.5} seconds.')




if __name__ == '__main__':
    main()
