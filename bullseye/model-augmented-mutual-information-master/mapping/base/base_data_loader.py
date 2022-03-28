import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler


class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders
    """
    def __init__(self, dataset, batch_size, shuffle, validation_split, test_split, num_workers, seed=0, collate_fn=default_collate):
        self.validation_split = validation_split
        self.test_split = test_split
        self.shuffle = shuffle
        
        self.batch_idx = 0
        self.n_samples = len(dataset)

        self.sampler, self.valid_sampler, self.test_sampler = self._split_sampler(self.validation_split, self.test_split)
        self.seed = seed

        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers
            }
        super(BaseDataLoader, self).__init__(sampler=self.sampler, **self.init_kwargs)

    def _split_sampler(self, split, split_test):
        if split == 0.0 and split_test == 0:
            return None, None, None

        idx_full = np.arange(self.n_samples)

        np.random.seed(self.seed)
        np.random.shuffle(idx_full)

        len_valid = int(self.n_samples * split)
        len_test = int(self.n_samples * split_test)

        test_idx = idx_full[0:len_test]
        valid_idx = idx_full[len_test:len_test+len_valid]
        train_idx = np.delete(idx_full, np.arange(0, len_test+len_valid))

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        test_sampler = SubsetRandomSampler(test_idx)

        # save for later
        self.train_idx = train_idx
        self.valid_idx = valid_idx
        self.test_idx = test_idx

        # turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False
        self.n_samples = len(train_idx)

        return train_sampler, valid_sampler, test_sampler

    def split_validation(self):
        if self.valid_sampler is None:
            return None
        else:
            return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)

    def split_test(self):
        if self.test_sampler is None:
            return None
        else:
            return DataLoader(sampler=self.test_sampler, **self.init_kwargs)
