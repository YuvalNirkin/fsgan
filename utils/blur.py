import math
import numbers
import torch
from torch import nn
from torch.nn import functional as F


# Adapted from:
# https://discuss.pytorch.org/t/is-there-anyway-to-do-gaussian-filtering-for-an-image-2d-3d-in-pytorch/12351/10
class GaussianSmoothing(nn.Module):
    """ Apply gaussian smoothing on a 1d, 2d or 3d tensor.

    Filtering is performed seperately for each channel in the input using a depthwise convolution.

    Args:
        channels (int): Number of channels for both input and output tensors.
        kernel_size(int or list of int): Size of the gaussian kernel
        sigma (float or list of float): Standard deviation of the gaussian kernel
        padding (int, optional): Padding size. The default is half the kernel size
        dim (int, optional): The number of dimensions of the data. Default value is 2 (spatial)
    """
    def __init__(self, channels, kernel_size, sigma, padding=None, dim=2):
        super(GaussianSmoothing, self).__init__()
        self.padding = kernel_size // 2 if padding is None else padding
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
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
                      torch.exp(-((mgrid - mean) / (2 * std)) ** 2)

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
        """ Apply gaussian filter to input.

        Args:
            input (torch.Tensor): Input to apply gaussian filter on.

        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups, padding=self.padding)
