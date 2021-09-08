import torch

class Intensity2Latency:
	r"""Applies intensity to latency transform. Spike waves are generated in the form of
	spike bins with almost equal number of spikes.
	Args:
		number_of_spike_bins (int): Number of spike bins (time steps).
		to_spike (boolean, optional): To generate spike-wave tensor or not. Default: False
	.. note::
		If :attr:`to_spike` is :attr:`False`, then the result is intesities that are ordered and packed into bins.
	"""
	def __init__(self, number_of_spike_bins, to_spike=False):
		self.time_steps = number_of_spike_bins
		self.to_spike = to_spike
	
	# intencities is a tensor of input intencities (1, input_channels, height, width)
	# returns a tensor of tensors containing spikes in each timestep (considers minibatch for timesteps)
	# spikes are accumulative, i.e. spikes in timestep i are also presented in i+1, i+2, ...
	def intensity_to_latency(self, intencities):
		#bins = []
		bins_intencities = []
		nonzero_cnt = torch.nonzero(intencities, as_tuple=False).size()[0]

		#check for empty bins
		bin_size = nonzero_cnt//self.time_steps

		#sort
		intencities_flattened = torch.reshape(intencities, (-1,))
		intencities_flattened_sorted = torch.sort(intencities_flattened, descending=True)

		#bin packing
		sorted_bins_value, sorted_bins_idx = torch.split(intencities_flattened_sorted[0], bin_size), torch.split(intencities_flattened_sorted[1], bin_size)

		#add to the list of timesteps
		spike_map = torch.zeros_like(intencities_flattened_sorted[0])
	
		for i in range(self.time_steps):
			spike_map.scatter_(0, sorted_bins_idx[i], sorted_bins_value[i])
			spike_map_copy = spike_map.clone().detach()
			spike_map_copy = spike_map_copy.reshape(tuple(intencities.shape))
			bins_intencities.append(spike_map_copy.squeeze(0).float())
			#bins.append(spike_map_copy.sign().squeeze_(0).float())
	
		return torch.stack(bins_intencities)#, torch.stack(bins)
		#return torch.stack(bins)

	def __call__(self, image):
		if self.to_spike:
			return self.intensity_to_latency(image).sign()
		return self.intensity_to_latency(image)


