import h5py, torch, cv2
import numpy as np
from PIL import Image
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler


class CacheDataset(torch.utils.data.Dataset):
    r"""A wrapper dataset to cache pre-processed data. It can cache data on RAM or a secondary memory.
    .. note::
        Since converting image into spike-wave can be time consuming, we recommend to wrap your dataset into a :attr:`CacheDataset`
        object.
    Args:
        dataset (torch.utils.data.Dataset): The reference dataset object.
        cache_address (str, optional): The location of cache in the secondary memory. Use :attr:`None` to cache on RAM. Default: None
    """
    def __init__(self, dataset, cache_address=None):
        self.dataset = dataset
        self.cache_address = cache_address
        self.cache = [None] * len(self.dataset)

    def __getitem__(self, index):
        if self.cache[index] is None:
            #cache it
            sample, target = self.dataset[index]
            if self.cache_address is None:
                self.cache[index] = sample, target
            else:
                save_path = os.path.join(self.cache_address, str(index))
                torch.save(sample, save_path + ".cd")
                torch.save(target, save_path + ".cl")
                self.cache[index] = save_path
        else:
            if self.cache_address is None:
                sample, target = self.cache[index]
            else:
                sample = torch.load(self.cache[index] + ".cd")
                target = torch.load(self.cache[index] + ".cl")
        return sample, target

    def reset_cache(self):
        r"""Clears the cached data. It is useful when you want to change a pre-processing parameter during
        the training process.
        """
        if self.cache_address is not None:
            for add in self.cache:
                os.remove(add + ".cd")
                os.remove(add + ".cl")
        self.cache = [None] * len(self)

    def __len__(self):
        return len(self.dataset)


def load_dataset(path, transform):

    trainsetfolder = CacheDataset(torch.utils.data.ImageFolder(path, transform))
    trainsetfolder, testsetfolder = torch.utils.data.random_split(trainsetfolder, 
                            [int(len(trainsetfolder) * 0.8), len(trainsetfolder) - int(len(trainsetfolder) * 0.8)])
    trainset = torch.utils.data.DataLoader(trainsetfolder, batch_size = len(trainsetfolder), shuffle = False)
    testset = torch.utils.data.DataLoader(testsetfolder, batch_size = len(testsetfolder), shuffle = False)
    return trainset, testset