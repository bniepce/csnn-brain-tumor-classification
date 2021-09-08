import cv2
import numpy as np
from skimage import color
from torchvision import transforms
from .encoding import Intensity2Latency
from utils import functional as sf
from dataset.filters import ADF
from PIL import Image

class InputTransform:
    def __init__(self, filter, timesteps = 15):
        self.to_tensor = transforms.ToTensor()
        self.filter = filter
        if self.filter.use_abs == True:
            self.temporal_transform = Intensity2Latency(timesteps, to_spike=True)
        else:
            self.temporal_transform = Intensity2Latency(timesteps)
        self.cnt = 0
        
    def __call__(self, image):
        image = self.to_tensor(image) * 255
        image.unsqueeze_(0)
        image = self.filter(image)
        if self.filter.use_abs == True:
            image = sf.pointwise_inhibition(image)
        else:
            image = sf.local_normalization(image, 8)
        temporal_image = self.temporal_transform(image)
        return temporal_image

class DeepTransform:
    def __init__(self, filter, timesteps = 15):
        self.to_tensor = transforms.ToTensor()
        self.grayscale = transforms.Grayscale()
        self.filter = filter
        self.temporal_transform = Intensity2Latency(timesteps)
        self.cnt = 0

    def apply_adf(self, image):
        im = image.convert('RGB') 
        open_cv_image = np.array(im) 
        open_cv_image = open_cv_image[:, :, ::-1].copy() 
        im = ADF(num_iter=4,delta_t=1/7,kappa=30,option=2).fit(open_cv_image)
        im = cv2.normalize(im, None, 0, 255, cv2.NORM_MINMAX)
        im = Image.fromarray(im)
        return im

    def __call__(self, image):
        image = self.apply_adf(image)
        image = self.to_tensor(self.grayscale(image)) * 255
        image.unsqueeze_(0)
        image = self.filter(image)
        image = sf.local_normalization(image, 8)
        temporal_image = self.temporal_transform(image)
        return temporal_image.sign().byte()


class ShallowTransform:
    def __init__(self, filter, pooling_size = 5, pooling_stride = 4, lateral_inhibition = None, timesteps = 15,
    feature_wise_inhibition=True):
        self.grayscale = transforms.Grayscale()
        self.to_tensor = transforms.ToTensor()
        self.filter = filter
        self.pooling_size = pooling_size
        self.pooling_stride = pooling_stride
        self.lateral_inhibition = lateral_inhibition
        self.temporal_transform = Intensity2Latency(timesteps)
        self.feature_wise_inhibition = feature_wise_inhibition

    def apply_adf(self, image):
        im = image.convert('RGB') 
        open_cv_image = np.array(im) 
        open_cv_image = open_cv_image[:, :, ::-1].copy() 
        im = ADF(num_iter=4,delta_t=1/7,kappa=30,option=2).fit(open_cv_image)
        im = cv2.normalize(im, None, 0, 255, cv2.NORM_MINMAX)
        im = Image.fromarray(im)
        return im

    def __call__(self, image):
        image = self.apply_adf(image)
        image = self.to_tensor(self.grayscale(image))
        image.unsqueeze_(0)
        image = self.filter(image)
        image = sf.pooling(image, self.pooling_size, self.pooling_stride, padding=self.pooling_size//2)
        if self.lateral_inhibition is not None:
            image = self.lateral_inhibition(image)
        temporal_image = self.temporal_transform(image)
        temporal_image = sf.pointwise_inhibition(temporal_image)
        return temporal_image.sign().byte()