import torch
from kornia.core import Tensor, Module, Parameter

class TransformLayer(Module):
    """
    Module to normalize an input tensor using mean and standard deviation.

    Args:
        apply (bool): Whether to apply normalization or not.
        mean (List[float]): Mean values used for normalization.
        std (List[float]): Standard deviation values used for normalization.

    Returns:
        Tensor: Normalized tensor.

    Shepe:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H, W)`
    """

    def __init__(self, apply: Tensor, mean, std):
        super().__init__()
        self.apply = apply
        if self.apply:
            mean = torch.as_tensor(mean, dtype=torch.float)[None, :, None, None]
            std = torch.as_tensor(std, dtype=torch.float)[None, :, None, None]
            self.mean = Parameter(mean, requires_grad=False)
            self.std = Parameter(std, requires_grad=False)

    def forward(self, x: Tensor):
        if self.apply:
            return x.sub(self.mean).div(self.std)
        else:
            return x