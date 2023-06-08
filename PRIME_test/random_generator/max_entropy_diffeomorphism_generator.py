from typing import Dict, Tuple

import torch
import math
from kornia.augmentation.random_generator.base import RandomGeneratorBase
from kornia.augmentation.utils import (
    _common_param_check,
    _adapted_rsampling,
)
from kornia.core import Tensor
from kornia.utils.helpers import _extract_device_dtype
from torch.distributions import Uniform, Beta, Normal

__all__ = ["MaxEntropyDiffeomorphismGenerator"]


class MaxEntropyDiffeomorphismGenerator(RandomGeneratorBase):
    r"""This class generates the parameters for the Maximum Entropy Diffeomorphism class.

    This transform follows the theroetical framework of PRIME,
    maximizing the entropy of the output image.
    To read more about the theory behind this transformation, please refer to the paper
    `PRIME: A Few Primitives Can Boost Robustness to Common Corruptions <https://arxiv.org/abs/2112.13547>`.


    Args:
        iT (float): Temperature parameter for controlling the amount of diffeomorphism.
        jT (float): Temperature parameter for controlling the amount of Jacobian determinant.
        i_max (float): Upper limit of iT.
        j_max (float): Upper limit of jT.
        k_min (float): Minimum allowed value for the range of `cut` frequencies.
        k_max (float): Maximum allowed value for the range of `cut` frequencies.
        sigma_max (float): Maximum standard deviation value for the strength distribution.

    Returns:
        Dict[str, Tensor]: a dictionary containing the temperature parameter `T` and the cut value `cut`.

    """

    def __init__(self, iT, jT, i_max, j_max, k_min, k_max, sigma_max):
        super().__init__()

        self.iT = iT
        self.jT = jT
        self.i_max = i_max
        self.j_max = j_max
        self.k_min = k_min
        self.k_max = k_max
        self.sigma_max = sigma_max

    def make_samplers(self, device: torch.device, dtype: torch.dtype) -> None:
        self.betaT_sampler = Beta(
            self.iT - self.iT / (self.jT + 1),
            self.iT / (self.jT + 1),
            validate_args=None,
        )
        self.betacut_sampler = Beta(
            self.i_max - self.i_max / (self.j_max + 1),
            self.i_max / (self.j_max + 1),
            validate_args=None,
        )
        self.k_max_sampler = Uniform(
            low=self.k_min + 1, high=self.k_max + 1, validate_args=False
        )
        self.sigma_sampler = Uniform(low=0, high=self.sigma_max, validate_args=False)
        self.coeffs_sampler = Normal(loc=0.0, scale=1.0, validate_args=False)

    def temperature_range(self, n, cut):
        """
        Define the range of allowed temperature
        for given image size and cut.
        """
        if cut == 0:
            print("Cut is zero!")
        if isinstance(cut, (float, int)):
            cut = cut + 1e-6
            log = math.log(cut)
        else:
            log = cut.log()
        T1 = 1 / (math.pi * n**2 * log)
        T2 = 4 / (math.pi**3 * cut**2 * log)
        return T1, T2

    def forward(
        self, batch_shape: Tuple[int, ...], same_on_batch: bool = False
    ) -> Dict[str, Tensor]:
        _common_param_check(batch_shape[0], same_on_batch)
        _device, _dtype = _extract_device_dtype(
            [
                self.iT,
                self.jT,
                self.i_max,
                self.j_max,
                self.k_min,
                self.k_max,
                self.sigma_max,
            ]
        )
        # Get image size from batch shape
        n = batch_shape[-1]

        # Sample values for sigma, betacut, and betaT
        sigma = _adapted_rsampling((1,), self.sigma_sampler, same_on_batch).item()
        betacut = _adapted_rsampling((1,), self.betacut_sampler, same_on_batch).item()
        betaT = _adapted_rsampling((1,), self.betaT_sampler, same_on_batch).item()

        # Calculate temperature and cut values
        cut = betacut * (self.k_max + 1 - self.k_min) + self.k_min
        cut = torch.tensor(cut, dtype=torch.int)
        T1, T2 = self.temperature_range(n, cut)
        T2 = max(T1, sigma * T2)
        T = betaT * (T2 - T1) + T1
        T = torch.tensor(T, dtype=torch.float)

        coeffs = _adapted_rsampling(
            (batch_shape[0], cut, cut), self.coeffs_sampler, same_on_batch
        )

        return dict(
            T=T.to(device=_device, dtype=_dtype),
            cut=cut.to(device=_device, dtype=_dtype),
            coeffs=coeffs.to(device=_device, dtype=_dtype),
        )
