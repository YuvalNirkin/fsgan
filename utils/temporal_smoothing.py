import torch
from torch import nn
from torch.nn import functional as F


class TemporalSmoothing(nn.Module):
    """ Apply temporal smoothing kernel on the batch dimension of a 4d tensor.

    Filtering is performed separately for each channel in the input using a depthwise convolution.

    Args:
        channels (int): Number of channels of the input tensors. Output will
            have this number of channels as well
        kernel_size (int): Size of the average kernel
    """
    def __init__(self, channels, kernel_size=5):
        super(TemporalSmoothing, self).__init__()
        self.kernel_size = kernel_size
        self.kernel_radius = kernel_size // 2
        self.groups = channels

        # Create kernel
        kernel = torch.ones(channels, 1, kernel_size, 1)
        kernel.div_(kernel_size)
        self.register_buffer('weight', kernel)

    def forward(self, x, pad_prev=0, pad_next=0):
        """ Apply temporal smoothing to x.

        Args:
            x (torch.Tensor): Input to apply temporal smoothing on
            pad_prev (int): The amount of reflection padding from the left side of the batch dimension
            pad_next (int): The amount of reflection padding from the right side of the batch dimension

        Returns:
            torch.Tensor: Filtered output.
        """
        orig_shape = x.shape

        # Transform tensor for temporal filtering
        x = x.permute(1, 0, 2, 3)
        x = x.view(1, x.shape[0], x.shape[1], -1)
        if pad_prev > 0 or pad_next > 0:
            x = F.pad(x, (0, 0, pad_prev, pad_next), 'reflect')

        # Apply temporal convolution
        x = F.conv2d(x, self.weight, groups=self.groups)

        # Transform tensor back to original shape
        x = x.permute(0, 2, 1, 3)
        x = x.view((x.shape[1],) + orig_shape[1:])

        return x


def smooth_temporal(x, kernel_size=5, pad_prev=0, pad_next=0):
    """ Apply dynamic temporal smoothing kernel on the batch dimension of a 4d tensor.

    Filtering is performed separately for each channel in the input using a depthwise convolution.

    Args:
        x (torch.Tensor): Input to apply temporal smoothing on
        kernel_size (int): Size of the average kernel
        pad_prev (int): The amount of reflection padding from the left side of the batch dimension
        pad_next (int): The amount of reflection padding from the right side of the batch dimension

    Returns:
        torch.Tensor: Filtered output.
    """
    orig_shape = x.shape

    # Create kernel
    kernel = torch.ones(x.shape[1], 1, kernel_size, 1).to(x.device)
    kernel.div_(kernel_size)

    # Transform tensor for temporal filtering
    x = x.permute(1, 0, 2, 3)
    x = x.view(1, x.shape[0], x.shape[1], -1)
    if pad_prev > 0 or pad_next > 0:
        x = F.pad(x, (0, 0, pad_prev, pad_next), 'reflect')

    # Apply temporal convolution
    x = F.conv2d(x, kernel, groups=x.shape[1])

    # Transform tensor back to original shape
    x = x.permute(0, 2, 1, 3)
    x = x.view((x.shape[1],) + orig_shape[1:])

    return x
