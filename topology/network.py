import torch
from . import layers as spiking_layers
from topology import learning
from abc import ABC, abstractmethod, abstractproperty
from torch.nn.parameter import Parameter
from utils import functional as sf
from utils import get_module_classes, count_layer_instance
from uuid import uuid4

class AbstractNetwork(ABC):

    def __init__(self, name : str):
        super().__init__()
        self.__init_name(name)
        self.max_ap = Parameter(torch.Tensor([0.15]))
        self.ctx = {"input_spikes": None, 
                    "potentials": None, 
                    "output_spikes": None, 
                    "winners": None}
        self.spike_counts = []
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
    
    def __init_name(self, name : str = None) -> None:
        if not name:
            prefix = self.__class__.__name__.lower()
            self.name = prefix + '_' + str(uuid4().hex[:5])
        else:
            self.name = name

    def save_spikes(self, input_spike, potentials, output_spikes, winners):
        self.ctx["input_spikes"] = input_spike
        self.ctx["potentials"] = potentials
        self.ctx["output_spikes"] = output_spikes
        self.ctx["winners"] = winners 
    
    @abstractmethod
    def add_layer(self) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def add_learning_rule(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def forward(self) -> None:
        raise NotImplementedError


class Network(torch.nn.Module, AbstractNetwork):
    def __init__(self, input_channels : int, features_per_class : int, num_classes : int,
                ks : list, inh_radiuses : list, conv_t : list, name : str) -> None:
        torch.nn.Module.__init__(self)
        AbstractNetwork.__init__(self, name)
        self.ks = ks
        self.inh_radiuses = inh_radiuses
        self.conv_t = conv_t
        self.num_classes = num_classes
        self.conv_layer_list = []
        self.pool_layer_list = []
        self.rule_list = []
        self.input_channels = input_channels
        self.features_per_class = features_per_class
        self.output_channels = self.features_per_class * self.num_classes


    def add_layer(self, layer, name=None) -> None:
        """
        Parameters
        ----------
        layer : Layer
            the layer to add to the network
        Raises
        ----------
        TypeError
            If the object passed to the function is not
            an instance of AbstractLayer
        """
        layers = get_module_classes(spiking_layers)
        if not isinstance(layer, tuple(layers)):
            raise TypeError('The added layer must be '
                            'an instance of class nn.Module. '
                            'Found: ' + str(layer))
        else:
            layer_count = count_layer_instance(self, layer.__class__)
            if isinstance(layer, spiking_layers.Convolution):
                prefix = 'conv'
                l = self.conv_layer_list
                self.spike_counts.append(0)
            else:
                l = self.pool_layer_list
                prefix = 'pool'

            layer_name = name if name is not None else '{}{}'.format(prefix, layer_count + 1)
            l.append(layer_name)
            self.add_module(name=layer_name, module=layer)
            return getattr(self, layer_name)

    def add_learning_rule(self, rule, learning_rates, to_layer, use_stabilizer = True, lower_bound = 0, upper_bound = 1, name=None) -> None:
        
        learning_classes = get_module_classes(learning)
        matches = [n.match(rule) for n in learning_classes]
        for i in matches:
            if i is not None:
                learning_instance = i
            else:
                raise ValueError('{} is not a valid learning rule.'.format(learning_rule))
        
        learning_rule = learning_instance(conv_layer=to_layer, 
                                learning_rate=learning_rates,
                                use_stabilizer = use_stabilizer, 
                                lower_bound = lower_bound, 
                                upper_bound = upper_bound)
        rule_count = count_layer_instance(self, learning_rule.__class__)
        rule_name = name if name is not None else 'learning_rule{}'.format(rule_count + 1)
        self.rule_list.append(rule_name)
        self.add_module(name = rule_name, module = learning_rule)
        return

    def compile_learning_rules(self, layer_idx):
        if layer_idx != 0:
            rule = getattr(self, 'learning_rule{}'.format(layer_idx))
            rule(self.ctx["input_spikes"], self.ctx["potentials"], self.ctx["output_spikes"], self.ctx["winners"])

    def get_winning_spike(self, spk_in, pot, spk, layer_id, update=True):
        if update:
            self.spike_counts[layer_id - 1] += 1
            rule = getattr(self, 'learning_rule{}'.format(layer_id))
            if self.spike_counts[layer_id - 1] >= 500:
                self.spike_counts[layer_id - 1] = 0
                ap = torch.tensor(rule.learning_rate[0][0].item(), \
                    device=rule.learning_rate[0][0].device) * 2
                ap = torch.min(ap, self.max_ap)
                an = ap * -0.75
                rule.update_all_learning_rate(ap.item(), an.item())
        pot = sf.pointwise_inhibition(pot)
        spk = pot.sign()
        winners = sf.get_k_winners(pot, self.ks[layer_id - 1], self.inh_radiuses[layer_id - 1], spk)
        self.save_spikes(spk_in, pot, spk, winners)
        return spk, pot
    
    def compute_pool_layer(self, spk_in, layer_id, pad=(1,1,1,1)):
        if 'pool{}'.format(layer_id) in self.pool_layer_list:
            p = getattr(self, 'pool{}'.format(layer_id))
            spk_in = sf.pad(p(spk_in), pad) if pad is not None else p(spk_in)
        return spk_in

    def compute_conv_layer(self, spk_in, layer_id):
        conv_layer = getattr(self, 'conv{}'.format(layer_id))
        pot = conv_layer(spk_in)
        spk, pot = sf.fire(pot, self.conv_t[layer_id - 1], True)
        return spk, pot

    def forward(self, input, current_layer_id):
        spk_in = sf.pad(input.float(), (2,2,2,2), 0)
        last_layer_id = len(self.conv_layer_list)
        if self.training:
            spk, pot = self.compute_conv_layer(spk_in, 1)
            if current_layer_id == 1:
                self.spike_counts[0] += 1
                if self.spike_counts[0] >= 500:
                    self.spike_counts[0] = 0
                    rule = getattr(self, 'learning_rule{}'.format(current_layer_id))
                    ap = torch.tensor(rule.learning_rate[0][0].item(), 
                                    device=rule.learning_rate[0][0].device) * 2
                    ap = torch.min(ap, self.max_ap)
                    an = ap * -0.75
                    rule.update_all_learning_rate(ap.item(), an.item())
                pot = sf.pointwise_inhibition(pot)
                spk = pot.sign()
                winners = sf.get_k_winners(pot, self.ks[0], self.inh_radiuses[0], spk)
                self.save_spikes(spk_in, pot, spk, winners)
                return spk, pot
            spk_in = self.compute_pool_layer(spk, 1)
            spk_in = sf.pointwise_inhibition(spk_in)

            spk, pot = self.compute_conv_layer(spk_in, 2)
            if current_layer_id == 2:
                pot = sf.pointwise_inhibition(pot)
                spk = pot.sign()
                winners = sf.get_k_winners(pot, self.ks[1], self.inh_radiuses[1], spk)
                self.save_spikes(spk_in, pot, spk, winners)
                return spk, pot
            spk_out = self.compute_pool_layer(spk, 2, pad=None)
            return spk_out
        else:
            for idx in range(1, last_layer_id + 1):
                spk, pot = self.compute_conv_layer(spk_in, idx)
                if idx == last_layer_id:
                    spk_in = self.compute_pool_layer(spk_in, idx, None)
                else:
                    spk_in = self.compute_pool_layer(spk_in, idx)
                return spk_in

class RSTDP_Network(Network):

    def __init__(self, input_channels : int, features_per_class : int, num_classes : int, 
                ks : list,  inh_radiuses : list, conv_t : list, name : str = None):
        Network.__init__(self, input_channels, features_per_class, num_classes, ks, inh_radiuses, conv_t, name)
        self.decision_map = []
        for i in range(self.num_classes):
            self.decision_map.extend([i]*self.features_per_class)

    def update_learning_rates(self, stdp, anti_stdp, stdp_ap, stdp_an, anti_stdp_ap, anti_stdp_an):
        stdp.update_all_learning_rate(stdp_ap, stdp_an)
        anti_stdp.update_all_learning_rate(anti_stdp_an, anti_stdp_ap)

    def reward(self, stdp):
        stdp(self.ctx["input_spikes"], self.ctx["potentials"], self.ctx["output_spikes"], self.ctx["winners"])

    def punish(self, anti_stdp):
        anti_stdp(self.ctx["input_spikes"], self.ctx["potentials"], self.ctx["output_spikes"], self.ctx["winners"])
    
    def compute_output(self, spk_in, last_layer_id):
        pot = getattr(self, 'conv{}'.format(last_layer_id))(spk_in)
        spk = sf.fire(pot)
        winners = sf.get_k_winners(pot, 1, 0, spk)
        self.save_spikes(spk_in, pot, spk, winners)
        output = -1
        if len(winners) != 0:
            output = self.decision_map[winners[0][0]]
        return output

    def forward(self, input, max_layer):
        spk_in = sf.pad(input.float(), (2,2,2,2), 0)
        if self.training:
            spk, pot = self.compute_conv_layer(spk_in, 1)
            if max_layer == 1:
                spk, pot = self.get_winning_spike(spk_in, pot, spk, max_layer, True)
                return spk, pot
            spk_in = self.compute_pool_layer(spk, 1)
            spk, pot = self.compute_conv_layer(spk_in, 2)
            if max_layer == 2:
                spk, pot = self.get_winning_spike(spk_in, pot, spk, max_layer, True)
                return spk, pot
            spk_in = self.compute_pool_layer(spk, 2, (2, 2, 2, 2))
            output = self.compute_output(spk_in, 3)
            return output
        else:
            spk, pot = self.compute_conv_layer(spk_in, 1)
            if max_layer == 1:
                return spk, pot
            spk_in = self.compute_pool_layer(spk, 1)

            spk, pot = self.compute_conv_layer(spk_in, 2)
            if max_layer == 2:
                return spk, pot
            spk_in = self.compute_pool_layer(spk, 2, (2, 2, 2, 2))
            output = self.compute_output(spk_in, 3)
            return output
    # def forward(self, input, current_layer_id):
    #     spk_in = sf.pad(input.float(), (2,2,2,2), 0)
    #     last_layer_id = len(self.conv_layer_list)
    #     if self.training:
    #         if current_layer_id == last_layer_id:
    #             for idx in range(1, current_layer_id):
    #                 spk, pot = self.compute_conv_layer(spk_in, idx)
    #                 spk_in = self.compute_pool_layer(spk, idx)
    #             spk, pot = self.compute_conv_layer(spk_in, last_layer_id)
    #             output = self.compute_output(spk_in, last_layer_id)
    #             return output
    #         else:
    #             for idx in range(1, current_layer_id + 1):
    #                 spk, pot = self.compute_conv_layer(spk_in, idx)
    #                 spk_in = self.compute_pool_layer(spk, idx)
    #             spk, pot = self.get_winning_spike(spk_in, pot, spk, current_layer_id)
    #             return spk, pot
    #     else:
    #         for idx in range(1, last_layer_id):
    #             spk, pot = self.compute_conv_layer(spk_in, idx)
    #             if idx == current_layer_id:
    #                 return spk, pot
    #             else:
    #                 spk = self.compute_pool_layer(spk, idx)
    #         spk_in = self.compute_pool_layer(spk, current_layer_id, (2, 2, 2, 2))
    #         output = self.compute_output(spk_in, current_layer_id)
    #         return output