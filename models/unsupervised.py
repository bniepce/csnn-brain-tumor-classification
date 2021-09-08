from topology.network import Network
from topology.layers import Convolution, Pooling

class DeepSTDP(Network):
    def __init__(self, input_channels=6, features_per_class=10, num_classes=10, 
                ks=[5, 8], inh_radiuses=[2, 1], conv_t=[10, 1], name='KheradpishehMNIST'):
        Network.__init__(self, input_channels, features_per_class, num_classes, ks, inh_radiuses, conv_t, name)
        self.add_layer(layer=Convolution(self.input_channels, 32, 5, 0.8, 0.05))
        self.add_layer(layer=Pooling(2, 2))
    
        self.add_layer(layer=Convolution(32, self.output_channels, 3, 0.8, 0.05))
        self.add_layer(layer=Pooling(2, 2))

        self.add_learning_rule(rule='STDP', learning_rates=(0.004, -0.003), to_layer=self.conv1)
        self.add_learning_rule(rule='STDP', learning_rates=(0.004, -0.003), to_layer=self.conv2)