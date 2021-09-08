import torch, os
from uuid import uuid4
from abc import ABC, abstractmethod

class AbstractEstimator(object):
    def __init__(self, network, save_path):
        self.network = network
        self.save_path = save_path
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.type = 'abstract'
        self.model_name = '{}_spiking_model_{}'.format(self.type, str(uuid4().hex[:5]))
        os.makedirs(os.path.join(self.save_path, self.model_name), exist_ok=True)
    
    @abstractmethod
    def train(self):
        raise NotImplementedError
    
    @abstractmethod
    def test(self):
        raise NotImplementedError

    @abstractmethod
    def eval(self):
        raise NotImplementedError
    
