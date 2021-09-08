import torch
import torch.nn as nn
import torch.nn.functional as fn
from utils import functional as sf
from torch.nn.parameter import Parameter
from utils import to_pair

class Convolution(nn.Module):
	r"""Performs a 2D convolution over an input spike-wave composed of several input
	planes. Current version only supports stride of 1 with no padding.

	The input is a 4D tensor with the size :math:`(T, C_{{in}}, H_{{in}}, W_{{in}})` and the crresponsing output
	is of size :math:`(T, C_{{out}}, H_{{out}}, W_{{out}})`, 
	where :math:`T` is the number of time steps, :math:`C` is the number of feature maps (channels), and
	:math:`H`, and :math:`W` are the hight and width of the input/output planes.

	* :attr:`in_channels` controls the number of input planes (channels/feature maps).

	* :attr:`out_channels` controls the number of feature maps in the current layer.

	* :attr:`kernel_size` controls the size of the convolution kernel. It can be a single integer or a tuple of two integers.

	* :attr:`weight_mean` controls the mean of the normal distribution used for initial random weights.

	* :attr:`weight_std` controls the standard deviation of the normal distribution used for initial random weights.

	.. note::

		Since this version of convolution does not support padding, it is the user responsibility to add proper padding
		on the input before applying convolution.

	Args:
		in_channels (int): Number of channels in the input.
		out_channels (int): Number of channels produced by the convolution.
		kernel_size (int or tuple): Size of the convolving kernel.
		weight_mean (float, optional): Mean of the initial random weights. Default: 0.8
		weight_std (float, optional): Standard deviation of the initial random weights. Default: 0.02
	"""
	def __init__(self, in_channels, out_channels, kernel_size, weight_mean=0.8, weight_std=0.02):
		super(Convolution, self).__init__()
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.kernel_size = to_pair(kernel_size)
		#self.weight_mean = weight_mean
		#self.weight_std = weight_std

		# For future use
		self.stride = 1
		self.bias = None
		self.dilation = 1
		self.groups = 1
		self.padding = 0

		# Parameters
		self.weight = Parameter(torch.Tensor(self.out_channels, self.in_channels, *self.kernel_size))
		self.weight.requires_grad_(False) # We do not use gradients
		self.reset_weight(weight_mean, weight_std)

	def reset_weight(self, weight_mean=0.8, weight_std=0.02):
		"""Resets weights to random values based on a normal distribution.

		Args:
			weight_mean (float, optional): Mean of the random weights. Default: 0.8
			weight_std (float, optional): Standard deviation of the random weights. Default: 0.02
		"""
		self.weight.normal_(weight_mean, weight_std)

	def load_weight(self, target):
		"""Loads weights with the target tensor.

		Args:
			target (Tensor=): The target tensor.
		"""
		self.weight.copy_(target)	

	def forward(self, input):
		return fn.conv2d(input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class Pooling(nn.Module):
	r"""Performs a 2D max-pooling over an input signal (spike-wave or potentials) composed of several input
	planes.

	.. note::

		Regarding the structure of the spike-wave tensors, application of max-pooling over spike-wave tensors results
		in propagation of the earliest spike within each pooling window.

	The input is a 4D tensor with the size :math:`(T, C, H_{{in}}, W_{{in}})` and the crresponsing output
	is of size :math:`(T, C, H_{{out}}, W_{{out}})`, 
	where :math:`T` is the number of time steps, :math:`C` is the number of feature maps (channels), and
	:math:`H`, and :math:`W` are the hight and width of the input/output planes.

	* :attr:`kernel_size` controls the size of the pooling window. It can be a single integer or a tuple of two integers.

	* :attr:`stride` controls the stride of the pooling. It can be a single integer or a tuple of two integers. If the value is None, it does pooling with full stride.

	* :attr:`padding` controls the amount of padding. It can be a single integer or a tuple of two integers.

	Args:
		kernel_size (int or tuple): Size of the pooling window
		stride (int or tuple, optional): Stride of the pooling window. Default: None
		padding (int or tuple, optional): Size of the padding. Default: 0
	"""
	def __init__(self, kernel_size, stride=None, padding=0):
		super(Pooling, self).__init__()
		self.kernel_size = to_pair(kernel_size)
		if stride is None:
			self.stride = self.kernel_size
		else:
			self.stride = to_pair(stride)
		self.padding = to_pair(padding)

		# For future use
		self.dilation = 1
		self.return_indices = False
		self.ceil_mode = False

	def forward(self, input):
		return sf.pooling(input, self.kernel_size, self.stride, self.padding)