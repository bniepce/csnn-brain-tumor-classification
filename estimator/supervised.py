import numpy as np
import os, torch, sys
from datetime import datetime
from tqdm import tqdm
from uuid import uuid4
from . import AbstractEstimator


class SupervisedEstimator(AbstractEstimator):

    def __init__(self, network, save_path, adaptive_min=0.2, adaptive_int=0.8):
        super().__init__(network, save_path)

        self.type = 'supervised'
        self.learning_rule = getattr(self.network, 'learning_rule{}'.format(len(self.network.rule_list) - 1))
        self.punishing_rule = getattr(self.network, 'learning_rule{}'.format(len(self.network.rule_list)))

        self.apr = self.learning_rule.learning_rate[0][0].item()
        self.anr = self.learning_rule.learning_rate[0][1].item()
        self.app = self.punishing_rule.learning_rate[0][1].item()
        self.anp = self.punishing_rule.learning_rate[0][0].item()

        self.adaptive_min = adaptive_min
        self.adaptive_int = adaptive_int
        self.apr_adapt = ((1.0 - 1.0 / self.network.num_classes) * self.adaptive_int + self.adaptive_min) * self.apr
        self.anr_adapt = ((1.0 - 1.0 / self.network.num_classes) * self.adaptive_int + self.adaptive_min) * self.anr
        self.app_adapt = ((1.0 / self.network.num_classes) * self.adaptive_int + self.adaptive_min) * self.app
        self.anp_adapt = ((1.0 / self.network.num_classes) * self.adaptive_int + self.adaptive_min) * self.anp

    def train(self, data, target=None, method='rl', layer_idx = 3):
        if method == 'rl':
            self.network.train()
            perf = np.array([0,0,0])
            for i in range(len(data)):
                data_in = data[i].to(self.device)
                target_in = target[i].to(self.device)
                d = self.network(data_in, layer_idx)
                if d != -1:
                    if d == target_in:
                        perf[0]+=1
                        self.network.reward(self.learning_rule)
                    else:
                        perf[1]+=1
                        self.network.punish(self.punishing_rule)
                else:
                    perf[2]+=1
            return perf/len(data)

        elif method == 'unsupervised':
            self.network.train()
            for i in range(len(data)):
                data_in = data[i].to(self.device)
                self.network(data_in, layer_idx)
                self.network.compile_learning_rules(layer_idx)
        else:
            raise ValueError('Parameter method should be wether set to "rl" or "unsupervised".')

    def test(self, data, target, layer_idx):
        self.network.eval()
        perf = np.array([0,0,0]) # correct, wrong, silence
        for i in range(len(data)):
            data_in = data[i].to(self.device)
            target_in = target[i].to(self.device)
            d = self.network(data_in, layer_idx)
            if d != -1:
                if d == target_in:
                    perf[0]+=1
                else:
                    perf[1]+=1
            else:
                perf[2]+=1
        return perf/len(data)
    
    def update_rates(self, perf):
        self.apr_adapt = self.apr * (perf[1] * self.adaptive_int + self.adaptive_min)
        self.anr_adapt = self.anr * (perf[1] * self.adaptive_int + self.adaptive_min)
        self.app_adapt = self.app * (perf[0] * self.adaptive_int + self.adaptive_min)
        self.anp_adapt = self.anp * (perf[0] * self.adaptive_int + self.adaptive_min)

    def eval(self, train_dataloader, test_dataloader, epochs):
        print('\033[94m'+'\nStarting SUPERVISED training :\n\033[0m')
        if len(epochs) != len(self.network.conv_layer_list):
            raise ValueError('Number of elements in epochs list should be equal to the number of layers in the network')
        t = int(datetime.now().timestamp())
        net = self.network
        max_layer_id = len(net.conv_layer_list) 
        for layer_id in range(1, max_layer_id + 1):
            print('\033[94m'+'\nTraining layer n°{} :\n'.format(layer_id)+'\033[0m')
            best_train = np.array([0.0,0.0,0.0,0.0])
            best_test = np.array([0.0,0.0,0.0,0.0])
            for e in range(epochs[layer_id - 1]):
                perf_train = np.array([0.0,0.0,0.0])
                if layer_id != max_layer_id:
                    with tqdm(total=len(train_dataloader), desc='Epoch {} : '.format(e)) as pbar:
                        for data, targets in train_dataloader:
                            self.train(data, None, 'unsupervised', layer_id)
                            pbar.update()
                    model_file_path = os.path.join(self.save_path, self.model_name, 'saved_{}_{}.net'.format(layer_id, t))
                    torch.save(net.state_dict(), model_file_path)
                else:
                    for data, targets in train_dataloader:
                        perf_train_batch = self.train(data, targets, 'rl', max_layer_id)
                        print(perf_train_batch)
                        self.update_rates(perf_train_batch)
                        stdp = getattr(net, 'learning_rule{}'.format(max_layer_id))
                        anti_stdp = getattr(net, 'learning_rule{}'.format(max_layer_id + 1))
                        net.update_learning_rates(stdp, anti_stdp, self.apr_adapt, self.anr_adapt, self.app_adapt, self.anp_adapt)
                        perf_train += perf_train_batch
                    perf_train /= len(train_dataloader)
                    if best_train[0] <= perf_train[0]:
                        best_train = np.append(perf_train, e)
                    print("Current Train:", perf_train)
                    print("   Best Train:", best_train)

                    for data,targets in test_dataloader:
                        perf_test = self.test(data, targets, layer_id)
                        if best_test[0] <= perf_test[0]:
                            best_test = np.append(perf_test, e)
                            model_file_path = os.path.join(self.save_path, self.model_name, 'saved_{}_{}.net'.format(layer_id, t))
                            torch.save(net.state_dict(), model_file_path)
                        print(" Current Test:", perf_test)
                        print("    Best Test:", best_test)
        print('\n\x1b[6;30;42m'+ 'All layers were successfully evaluated'+'\x1b[0m \n')
        return best_train, best_test