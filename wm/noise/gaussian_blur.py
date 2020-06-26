import torch
import torch.nn as nn
import numpy as np
import torchvision
from pathlib import Path
import torch.nn.functional as F

import os
import math
import numbers

import util

class GaussianBlur(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, sigma=1.0):
        super(GaussianBlur, self).__init__()
        channels = 3
        kernel_size = 5
        dim = 2

        kernel_size = [kernel_size] * dim
        
        self.sigma = sigma
        sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input. Input can be either a list of tensors -- [noised_image, cover_image], or a tensor -- noised_image 
        """
        if isinstance(input, list) or isinstance(input, tuple):
            input = input[0]
        return self.conv(input, weight=self.weight, groups=self.groups)
        
    def __repr__(self):
        return f'blur({self.sigma})'



def main():
    import os    
    from PIL import Image
    import torchvision.transforms.functional as TF
    from torchvision.transforms import CenterCrop

    gb = GaussianBlur(sigma=0.5)

    H, W = 256, 256
    center_crop = CenterCrop(size=(H, W))
    source_images = ['000000023533.jpg', '000000028124.jpg', '000000050004.jpg','000000050042.jpg', '000000111006.jpg']
    folder = '/home/ando/source/hidden/data/small/train/train_class'
    source_images = [os.path.join(folder, img) for img in source_images]
    cropped_images = []
    batch = torch.empty(size=(len(source_images), 3, H, W))
    for i, image in enumerate(source_images):
        image_pil = Image.open(image)
        image_pil = center_crop(image_pil)
        image_tensor = TF.to_tensor(np.array(image_pil))         
        image_tensor = image_tensor * 2 - 1  # transform from [0, 1] to [-1, 1]
        batch[i, :, :, :] = image_tensor

    batch_padded = F.pad(batch, (2, 2, 2, 2), mode='reflect')
    blurred_images = gb(batch_padded)
    print(blurred_images.min())
    print(blurred_images.max())
    folder = './test'
    Path(folder).mkdir(parents=True, exist_ok=True)
    
    util.common.save_images(original_images=batch, processed_images=blurred_images, filename=os.path.join(folder, 'test-blur.jpg'))    


if __name__ == "__main__":
    main()