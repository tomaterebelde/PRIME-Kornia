from typing import Dict, Tuple

import torch
from kornia.augmentation.random_generator.base import RandomGeneratorBase
from kornia.augmentation.utils import (
    _common_param_check,
    _adapted_rsampling,
)
from kornia.core import Tensor
from kornia.utils.helpers import _extract_device_dtype
from torch.distributions import Uniform, Normal

__all__ = ["MaxEntropyColorJitterGenerator"]


class MaxEntropyColorJitterGenerator(RandomGeneratorBase):
    r"""Generates random MaximumEntropyColorJitter parameters for a batch of images.

    Sampling is performed according to the distribution proposed in:
    `PRIME: A Few Primitives Can Boost Robustness to Common Corruptions <https://arxiv.org/abs/2112.13547>`.

    Args:
        k_max (int): Maximum number of frequency components to be used.
        sigma_max (float): Maximum standard deviation of the strength distribution.
        delta_bandwith (Optional[int]): Number of consecutive frequency components
            that define the transformation. If None, the bandwidth is set to k_max.
        sigma_min (float): Minimum standard deviation of the strength distribution. Defaults to 0.0.

    Returns:
        Dict[str, Tensor]: A dict of parameters to be passed for transformation.
            - k (torch.Tensor): The number of frequency components to be used.
            - coeffs (torch.Tensor): The coefficients of the frequency components.
    """

    def __init__(self, k_max, sigma_max, delta_bandwidth=None, sigma_min=0.0) -> None:
        super().__init__()
        self.k_max = k_max
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.delta_bandwidth = delta_bandwidth

    def __repr__(self) -> str:
        repr = (
            f"k_max={self.k_max}, sigma=({self.sigma_min}, {self.sigma_max}), delta_bandwidth="
            f"{self.delta_bandwidth}"
        )
        return repr

    def make_samplers(self, device: torch.device, dtype: torch.dtype) -> None:
        self.sigma_sampler = Uniform(
            low=self.sigma_min, high=self.sigma_max, validate_args=False
        )
        self.k_min_sampler = Uniform(low=1, high=self.k_max + 1, validate_args=False)
        self.coeffs_sampler = Normal(loc=0.0, scale=1.0, validate_args=False)

    def forward(
        self, batch_shape: Tuple[int, ...], same_on_batch: bool = False
    ) -> Dict[str, Tensor]:
        _common_param_check(batch_shape[0], same_on_batch)
        _device, _dtype = _extract_device_dtype(
            [self.k_max, self.delta_bandwidth, self.sigma_min, self.sigma_max]
        )

        # Sample sigma from the uniform distribution
        sigma = _adapted_rsampling((1,), self.sigma_sampler, same_on_batch).item()

        # Sample k_min from the uniform distribution if delta_bandwidth is not None
        if self.delta_bandwidth is not None:
            k_min = _adapted_rsampling((1,), self.k_min_sampler, same_on_batch).item()

            # Define k as a range of integers from k_min to min(k_min + delta_bandwidth, k_max + 1)
            k = torch.arange(
                k_min, min(k_min + self.delta_bandwidth, self.k_max + 1), device=_device
            )

            # Sample random coefficients from a normal distribution of shape (batch_size, channels, k.shape[0])
            coeffs = _adapted_rsampling(
                (
                    batch_shape[0],
                    batch_shape[1],
                    k.shape[0],
                ),
                self.coeffs_sampler,
                same_on_batch,
            )

        else:
            # Define k as a range of integers from 1 to k_max
            k = torch.arange(1, self.k_max + 1, device=_device)

            # Sample random coefficients from a normal distribution of shape (batch_size, channels, k_max)
            coeffs = _adapted_rsampling(
                (
                    batch_shape[0],
                    batch_shape[1],
                    self.k_max,
                ),
                self.coeffs_sampler,
                same_on_batch,
            )
        # Scale coefficients by the square root of sigma
        coeffs = coeffs * torch.sqrt(torch.tensor(sigma))

        # Return generated parameters as a dictionary with keys "rand_coeffs" and "k"
        return dict(
            rand_coeffs=coeffs.to(device=_device, dtype=_dtype),
            k=k.to(device=_device, dtype=_dtype),
        )
