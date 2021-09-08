import os, torchvision, torch, argparse
import numpy as np
from torch.utils.data import DataLoader, Subset
from torch.utils.data.dataset import random_split
from torchvision.datasets import ImageFolder
from dataset import CacheDataset
from dataset.input import DeepTransform
from dataset.filters import DoGKernel, GaborKernel, Filter
from models import DeepRSTDP
from estimator.supervised import SupervisedEstimator
from utils import write_perf_file
from utils.functional import LateralIntencityInhibition

if __name__ == "__main__":

	np.random.seed(1234)

	parser = argparse.ArgumentParser()
	parser.add_argument(
		'--data_path',
		type=str,
		required=True
	)
	parser.add_argument(
		'--save_path',
		type=str,
		default='./saved_models',
		required=True
	)
	parser.add_argument(
		'--epochs',
		nargs='+',
		required=True
	)
	parser.add_argument(
		'--task',
		default = 'ct3',
		required=True
	)
	parser.add_argument(
		'--input_filter',
		default = 'gabor',
		required=True
	)

	args = parser.parse_args()

	data_path = args.data_path
	task = args.task
	epochs = list(map(int, args.epochs))
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	os.makedirs(args.save_path, exist_ok = True)

	if task == 'ct1' or task == 'ct2':
		number_of_class = 2
	if task == 'ct3':
		number_of_class = 3
	else:
		raise ValueError('Task name not supported.')

	# LOADING DATASET
	if args.input_filter == 'dog':
		kernels = [DoGKernel(3,3/9,6/9),
			DoGKernel(3,6/9,3/9),
			DoGKernel(7,7/9,14/9),
			DoGKernel(7,14/9,7/9),
			DoGKernel(13,13/9,26/9),
			DoGKernel(13,26/9,13/9)]

		filter = Filter(kernels, padding = 6, thresholds = 50)
		s1c1 = DeepTransform(filter)
	else:
		kernels = [GaborKernel(5, 45+22.5),
				GaborKernel(5, 90+22.5),
				GaborKernel(5, 135+22.5),
				GaborKernel(5, 180+22.5)]

		filter = Filter(kernels, use_abs = True)
		lateral_inhibition = LateralIntencityInhibition([0.15, 0.12, 0.1, 0.07, 0.05])
		s1c1 = DeepTransform(filter)

	cache_dataset = CacheDataset(ImageFolder(data_path, s1c1))
	train_dataset, test_dataset = torch.utils.data.random_split(cache_dataset,
									[int(len(cache_dataset) * 0.8), 
									len(cache_dataset) - \
									int(len(cache_dataset) * 0.8)])
	#train_dataset, test_dataset = Subset(train_dataset, [0, 1, 2]), Subset(test_dataset, [0, 1, 2])
	# data_root = "./data"
	# number_of_class = 10
	# train_dataset = torchvision.datasets.MNIST(root=data_root, train=True, download=True, transform = s1c1)
	# test_dataset = torchvision.datasets.MNIST(root=data_root, train=False, download=True, transform = s1c1)
	# train_dataset = random_split(train_dataset, [10000, len(train_dataset)-10000])[0]
	# test_dataset = random_split(test_dataset, [10000, len(test_dataset)-10000])[0]
	train_dataloader = DataLoader(train_dataset, batch_size=len(train_dataset), pin_memory=True, num_workers=4, shuffle=False)
	test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), pin_memory=True, num_workers=4, shuffle=False)
	data_len = len(train_dataset)

	# TRAINING LOOP
	net = DeepRSTDP(input_channels=len(kernels), features_per_class=10, num_classes=number_of_class).to(device)
	estimator = SupervisedEstimator(net, save_path='./saved_models/')
	train_perf, test_perf = estimator.eval(train_dataloader=train_dataloader, 
										test_dataloader=test_dataloader,
										epochs=epochs)
	write_perf_file(net, task, data_len, estimator.type, kernels, epochs, train_perf, test_perf)