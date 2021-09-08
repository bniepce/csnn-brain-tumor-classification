import numpy as np
import os, torch, sys
from datetime import datetime
from tqdm import tqdm
from uuid import uuid4
from . import AbstractEstimator
from utils.functional import get_performance
from sklearn.svm import LinearSVC

class UnsupervisedEstimator(AbstractEstimator):

    def __init__(self, network, save_path):
        super().__init__(network, save_path)
        self.type = 'supervised'

    def train(self, data, layer_idx):
        self.network.train()
        for i in range(len(data)):
            data_in = data[i].to(self.device)
            self.network(data_in, layer_idx)
            self.network.compile_learning_rules(layer_idx)

    def test(self, data, target, layer_idx):
        self.network.eval()
        ans = [None] * len(data)
        t = [None] * len(data)
        for i in range(len(data)):
            data_in = data[i]
            data_in = data_in.to(self.device, dtype=torch.float)
            output,_ = self.network(data_in, layer_idx).max(dim = 0)
            ans[i] = output.reshape(-1).cpu().numpy()
            t[i] = target[i]
        return np.array(ans), np.array(t)

    def eval(self, train_dataloader, test_dataloader, epochs):
        if len(epochs) != len(self.network.conv_layer_list):
            raise ValueError('Number of elements in epochs list should be equal to the number of layers in the network')
        t = int(datetime.now().timestamp())
        net = self.network
        max_layer_id = len(net.conv_layer_list) 
        best_train = np.array([0.0,0.0,0.0,0.0])
        best_test = np.array([0.0,0.0,0.0,0.0])
        for layer_id in range(1, max_layer_id + 1):
            print('\033[94m'+'\nTraining layer n°{} :\n'.format(layer_id)+'\033[0m')
            model_file_path = os.path.join(self.save_path, self.model_name, 'saved_{}_{}.net'.format(layer_id, t))
            for e in range(epochs[layer_id - 1]):
                with tqdm(total=len(train_dataloader), desc='Epoch {} : '.format(e + 1)) as pbar:
                    for data, _ in train_dataloader:
                        self.train(data, layer_id)
                        pbar.update()
        print('\n\033[94m'+'Testing model with support vector classifier.'+'\033[0m \n')

        with tqdm(total=len(train_dataloader), desc='Evaluating classifier for training data : ') as pbar:
            for data, target in train_dataloader:
                train_X, train_y = self.test(data, target, max_layer_id)
                pbar.update()
        
        with tqdm(total=len(test_dataloader), desc='Evaluating classifier for testing data : ') as pbar:
            for data, target in test_dataloader:
                test_X, test_y = self.test(data, target, max_layer_id)
                pbar.update()

        clf = LinearSVC(C=2.4, verbose=0)
        clf.fit(train_X, train_y)
        predict_train = clf.predict(train_X)
        predict_test = clf.predict(test_X)

        g1 = get_performance(train_X, train_y, predict_train)
        g2 = get_performance(test_X, test_y, predict_test)
        print('\n\x1b[6;30;42m'+ 'All layers were successfully evaluated'+'\x1b[0m \n')
        return g1, g2
        