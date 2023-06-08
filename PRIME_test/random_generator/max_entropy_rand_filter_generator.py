from typing import Dict, List, Tuple
from math import ceil

import torch
from torch.distributions import Uniform, Normal

from kornia.augmentation.random_generator.base import RandomGeneratorBase
from kornia.augmentation.utils import _adapted_rsampling, _common_param_check
from kornia.core import Tensor
from kornia.utils.helpers import _extract_device_dtype

__all__ = ["MaxEntropyRandomFilterGenerator"]


class MaxEntropyRandomFilterGenerator(RandomGeneratorBase):
    r"""
    This class generates the parameters for the Maximum Entropy Random Filter class.

    This transform follows the theroetical framework of PRIME,
    maximizing the entropy of the output image.
    To read more about the theory behind this transformation, please refer to the paper
    `PRIME: A Few Primitives Can Boost Robustness to Common Corruptions <https://arxiv.org/abs/2112.13547>`.


    Args:
        kernel_size (int): Size of the kernel to be generated.
        sigma_max (float): Maximum standard deviation value for the strength distribution.
        sigma_min (float): Minimum standard deviation value for the strength distribution.

    Returns:
        Dict[str, Tensor]: a dictionary containing the convolution weights 'conv_weights'.

    """

    def __init__(
        self,
        kernel_size,
        sigma_max=0.1,
        sigma_min=0.01,
    ) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min

    def __repr__(self) -> str:
        repr = f"kernel_size={self.kernel_size}, sigma=({self.sigma_min}, {self.sigma_max})"
        return repr

    def make_samplers(self, device: torch.device, dtype: torch.dtype) -> None:
        self.kernels_size_candidates = torch.tensor(
            [float(i) for i in range(self.kernel_size, self.kernel_size + 2, 2)]
        )
        self.kernel_size = int(
            self.kernels_size_candidates[
                torch.multinomial(self.kernels_size_candidates, 1)
            ].item()
        )
        self.sigma_sampler = Uniform(
            self.sigma_min, self.sigma_max, validate_args=False
        )
        self.coeffs_sampler = Normal(loc=0.0, scale=1.0, validate_args=False)

    def _initialize_delta_filter(self, device):
        """Initializes the delta filter.

        Args:
            device (torch.device): device where the tensors will be allocated.

        Returns:
            Tensor: the delta filter.
        """

        delta = torch.zeros((1, self.kernel_size, self.kernel_size), device=device)
        center = int(ceil(self.kernel_size / 2))
        delta[0, center, center] = 1.0
        return delta

    def forward(
        self, batch_shape: Tuple[int, ...], same_on_batch: bool = False
    ) -> Dict[str, Tensor]:
        _common_param_check(batch_shape[0], same_on_batch)
        _device, _dtype = _extract_device_dtype(
            [self.kernel_size, self.sigma_max, self.sigma_min]
        )

        # Sample the parameters
        sigma = _adapted_rsampling((1,), self.sigma_sampler, same_on_batch).item()
        coeffs = _adapted_rsampling(
            (
                batch_shape[0],
                self.kernel_size,
                self.kernel_size,
            ),
            self.coeffs_sampler,
            same_on_batch,
        )
        delta = self._initialize_delta_filter(self.device)

        # Compute the convolution weights
        conv_weight = sigma * coeffs + delta
        conv_weight = conv_weight.reshape(
            batch_shape[0], 1, self.kernel_size, self.kernel_size
        )

        # Return generated parameters
        return dict(
            conv_weight=conv_weight.to(device=_device, dtype=_dtype),
        )
