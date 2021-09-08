from topology.layers import Convolution, Pooling
from topology.network import RSTDP_Network
from torch import nn

class DeepRSTDP(RSTDP_Network):
    def __init__(self, name : str = 'MozafariDeep', input_channels : int = 6, features_per_class : int = 10,
                    num_classes : int = 10, ks : list = [5, 8],
                    inh_radiuses : list = [3, 1], conv_t : list = [15, 10]):
        super().__init__(input_channels, features_per_class, num_classes, 
                        ks, inh_radiuses, conv_t, name)
        
        self.add_layer(layer=Convolution(self.input_channels, 64, 5, 0.8, 0.05))
        self.add_layer(layer=Pooling(2, 2))

        self.add_layer(layer=Convolution(64, 250, 3, 0.8, 0.05))
        self.add_layer(layer=Pooling(3, 3))

        self.add_layer(layer=Convolution(250, self.output_channels, 5, 0.8, 0.05))

        self.add_learning_rule(rule='STDP', learning_rates=(0.004, -0.003), to_layer=self.conv1)
        self.add_learning_rule(rule='STDP', learning_rates=(0.004, -0.003), to_layer=self.conv2)
        self.add_learning_rule(rule='STDP', learning_rates=(0.004, -0.003), 
                            use_stabilizer=False, lower_bound=0.2,
                            upper_bound=0.8, to_layer=self.conv3)
        self.add_learning_rule(rule='STDP', learning_rates=(-0.004, 0.0005), 
                            use_stabilizer=False, lower_bound=0.2,
                            upper_bound=0.8, to_layer=self.conv3)