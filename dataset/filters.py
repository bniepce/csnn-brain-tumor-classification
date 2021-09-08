import torch, math, cv2
import torch.nn.functional as fn
import numpy as np
import cv2
from PIL import Image
from skimage import color


class FilterKernel:
	r"""Base class for generating image filter kernels such as Gabor, DoG, etc. Each subclass should override :attr:`__call__` function.
	"""
	def __init__(self, window_size):
		self.window_size = window_size

	def __call__(self):
		pass

class DoGKernel(FilterKernel):
	r"""Generates DoG filter kernel.
	Args:
		window_size (int): The size of the window (square window).
		sigma1 (float): The sigma for the first Gaussian function.
		sigma2 (float): The sigma for the second Gaussian function.
	"""
	def __init__(self, window_size, sigma1, sigma2):
		super(DoGKernel, self).__init__(window_size)
		self.sigma1 = sigma1
		self.sigma2 = sigma2

	# returns a 2d tensor corresponding to the requested DoG filter
	def __call__(self):
		w = self.window_size//2
		x, y = np.mgrid[-w:w+1:1, -w:w+1:1]
		a = 1.0 / (2 * math.pi)
		prod = x*x + y*y
		f1 = (1/(self.sigma1*self.sigma1)) * np.exp(-0.5 * (1/(self.sigma1*self.sigma1)) * (prod))
		f2 = (1/(self.sigma2*self.sigma2)) * np.exp(-0.5 * (1/(self.sigma2*self.sigma2)) * (prod))
		dog = a * (f1-f2)
		dog_mean = np.mean(dog)
		dog = dog - dog_mean
		dog_max = np.max(dog)
		dog = dog / dog_max
		dog_tensor = torch.from_numpy(dog)
		return dog_tensor.float()

class GaborKernel(FilterKernel):
	r"""Generates Gabor filter kernel.
	Args:
		window_size (int): The size of the window (square window).
		orientation (float): The orientation of the Gabor filter (in degrees).
		div (float, optional): The divisor of the lambda equation. Default: 4.0
	"""
	def __init__(self, window_size, orientation, div=4.0):
		super(GaborKernel, self).__init__(window_size)
		self.orientation = orientation
		self.div = div

	# returns a 2d tensor corresponding to the requested Gabor filter
	def __call__(self):
		w = self.window_size//2
		x, y = np.mgrid[-w:w+1:1, -w:w+1:1]
		lamda = self.window_size * 2 / self.div
		sigma = lamda * 0.8
		sigmaSq = sigma * sigma
		g = 0.3
		theta = (self.orientation * np.pi) / 180
		Y = y*np.cos(theta) - x*np.sin(theta)
		X = y*np.sin(theta) + x*np.cos(theta)
		gabor = np.exp(-(X * X + g * g * Y * Y) / (2 * sigmaSq)) * np.cos(2 * np.pi * X / lamda)
		gabor_mean = np.mean(gabor)
		gabor = gabor - gabor_mean
		gabor_max = np.max(gabor)
		gabor = gabor / gabor_max
		gabor_tensor = torch.from_numpy(gabor)
		return gabor_tensor.float()

class Filter:
	r"""Applies a filter transform. Each filter contains a sequence of :attr:`FilterKernel` objects.
	The result of each filter kernel will be passed through a given threshold (if not :attr:`None`).
	Args:
		filter_kernels (sequence of FilterKernels): The sequence of filter kernels.
		padding (int, optional): The size of the padding for the convolution of filter kernels. Default: 0
		thresholds (sequence of floats, optional): The threshold for each filter kernel. Default: None
		use_abs (boolean, optional): To compute the absolute value of the outputs or not. Default: False
	.. note::
		The size of the compund filter kernel tensor (stack of individual filter kernels) will be equal to the 
		greatest window size among kernels. All other smaller kernels will be zero-padded with an appropriate 
		amount.
	"""
	# filter_kernels must be a list of filter kernels
	# thresholds must be a list of thresholds for each kernel
	def __init__(self, filter_kernels, padding=0, thresholds=None, use_abs=False):
		tensor_list = []
		self.max_window_size = 0
		for kernel in filter_kernels:
			if isinstance(kernel, torch.Tensor):
				tensor_list.append(kernel)
				self.max_window_size = max(self.max_window_size, kernel.size(-1))
			else:
				tensor_list.append(kernel().unsqueeze(0))
				self.max_window_size = max(self.max_window_size, kernel.window_size)
		for i in range(len(tensor_list)):
			p = (self.max_window_size - filter_kernels[i].window_size)//2
			tensor_list[i] = fn.pad(tensor_list[i], (p,p,p,p))

		self.kernels = torch.stack(tensor_list)
		self.number_of_kernels = len(filter_kernels)
		self.padding = padding
		if isinstance(thresholds, list):
			self.thresholds = thresholds.clone().detach()
			self.thresholds.unsqueeze_(0).unsqueeze_(2).unsqueeze_(3)
		else:
			self.thresholds = thresholds
		self.use_abs = use_abs

	# returns a 4d tensor containing the flitered versions of the input image
	# input is a 4d tensor. dim: (minibatch=1, filter_kernels, height, width)
	def __call__(self, input):
		output = fn.conv2d(input, self.kernels, padding = self.padding).float()
		if not(self.thresholds is None):
			output = torch.where(output < self.thresholds, torch.tensor(0.0, device=output.device), output)
		if self.use_abs:
			torch.abs_(output)
		return output

class ADF(object):

    def __init__(self,num_iter=10, delta_t=1/7, kappa=30,option=2):

        super(ADF,self).__init__()

        self.num_iter = num_iter
        self.delta_t = delta_t
        self.kappa = kappa
        self.option = option

        self.hN = np.array([[0,1,0],[0,-1,0],[0,0,0]])
        self.hS = np.array([[0,0,0],[0,-1,0],[0,1,0]])
        self.hE = np.array([[0,0,0],[0,-1,1],[0,0,0]])
        self.hW = np.array([[0,0,0],[1,-1,0],[0,0,0]])
        self.hNE = np.array([[0,0,1],[0,-1,0],[0,0,0]])
        self.hSE = np.array([[0,0,0],[0,-1,0],[0,0,1]])
        self.hSW = np.array([[0,0,0],[0,-1,0],[1,0,0]])
        self.hNW = np.array([[1,0,0],[0,-1,0],[0,0,0]])

    def fit(self,img):

        diff_im = img.copy()

        dx=1; dy=1; dd = math.sqrt(2)

        for i in range(self.num_iter):

            nablaN = cv2.filter2D(diff_im,-1,self.hN)
            nablaS = cv2.filter2D(diff_im,-1,self.hS)
            nablaW = cv2.filter2D(diff_im,-1,self.hW)
            nablaE = cv2.filter2D(diff_im,-1,self.hE)
            nablaNE = cv2.filter2D(diff_im,-1,self.hNE)
            nablaSE = cv2.filter2D(diff_im,-1,self.hSE)
            nablaSW = cv2.filter2D(diff_im,-1,self.hSW)
            nablaNW = cv2.filter2D(diff_im,-1,self.hNW)

            cN = 0; cS = 0; cW = 0; cE = 0; cNE = 0; cSE = 0; cSW = 0; cNW = 0

            if self.option == 1:
                cN = np.exp(-(nablaN/self.kappa)**2)
                cS = np.exp(-(nablaS/self.kappa)**2)
                cW = np.exp(-(nablaW/self.kappa)**2)
                cE = np.exp(-(nablaE/self.kappa)**2)
                cNE = np.exp(-(nablaNE/self.kappa)**2)
                cSE = np.exp(-(nablaSE/self.kappa)**2)
                cSW = np.exp(-(nablaSW/self.kappa)**2)
                cNW = np.exp(-(nablaNW/self.kappa)**2)
            elif self.option == 2:
                cN = 1/(1+(nablaN/self.kappa)**2)
                cS = 1/(1+(nablaS/self.kappa)**2)
                cW = 1/(1+(nablaW/self.kappa)**2)
                cE = 1/(1+(nablaE/self.kappa)**2)
                cNE = 1/(1+(nablaNE/self.kappa)**2)
                cSE = 1/(1+(nablaSE/self.kappa)**2)
                cSW = 1/(1+(nablaSW/self.kappa)**2)
                cNW = 1/(1+(nablaNW/self.kappa)**2)

            diff_im = diff_im + self.delta_t * (

                (1/dy**2)*cN*nablaN +
                (1/dy**2)*cS*nablaS +
                (1/dx**2)*cW*nablaW +
                (1/dx**2)*cE*nablaE +

                (1/dd**2)*cNE*nablaNE +
                (1/dd**2)*cSE*nablaSE +
                (1/dd**2)*cSW*nablaSW +
                (1/dd**2)*cNW*nablaNW
            )
        
        l = color.rgb2gray(diff_im)
        l = cv2.normalize(l, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        return l

def make_filter(kernels, ratio=9, type='dog', padding=6, thresholds=50):
    ks = []
    for i in kernels:
        ks.append(DoGKernel(i, i/ratio, (i*2)/ratio))
        ks.append(DoGKernel(i, i*2/ratio, i/ratio))
    filter = Filter(ks, padding=padding, thresholds=thresholds)
    return filter