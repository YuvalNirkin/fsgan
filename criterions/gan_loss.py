import torch.nn as nn


# Adapted from https://github.com/NVIDIA/pix2pixHD/blob/master/models/networks.py
class GANLoss(nn.Module):
    """ Defines the GAN loss as described in the paper:
    `"High-Resolution Image Synthesis and Semantic Manipulation with Conditional GANs"
    <https://arxiv.org/pdf/1711.11585.pdf>`_.

    Args:
        use_lsgan (bool): If True, the least squares version will be used
    """
    def __init__(self, use_lsgan=True):
        super(GANLoss, self).__init__()
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = pred.new_full(pred.shape, target_is_real)
                loss += self.loss(pred, target_tensor)
            return loss
        else:
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)
