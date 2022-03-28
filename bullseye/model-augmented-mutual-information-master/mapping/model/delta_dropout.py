""" Drop out random number of features from 1 to delta+1 """
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


class DeltaDropout(nn.Module):
    """
        Inverted dropout, except drops out entire dimensions with same probability
    """
    def __init__(self, delta):
        super().__init__()
        self.delta = delta

    def forward(self, z, enabled=True):
        """
            # input dimension: (N, dim, n_features)
            # choose random features to drop out entirely
        """
        if enabled:
            batch_size = z.shape[0]
            num_features = z.shape[2]

            mask = Variable(z.new(z.shape))
            mask[:] = 0
            for i in range(batch_size):
                c = np.random.choice(self.delta+1)+1
                s = np.sort(np.random.choice(num_features, c, replace=False))
                mask[i,:,s] = 1

            return mask*z

        else:
            return z
