from typing import Any, Dict, Optional

import torch
from kornia.augmentation._2d.intensity.base import IntensityAugmentationBase2D
from kornia.core import Tensor
from PRIME.random_generator.max_entropy_color_jitter_generator import (
    MaxEntropyColorJitterGenerator,
)

from einops import parse_shape, rearrange
from opt_einsum import contract


class MaxEntropyColorJitter(IntensityAugmentationBase2D):
    r"""Apply a maximum entropy random color jitter transformation to a tensor image.

    This is one of the three base transformations that defines PRIME. The strength of
    the transformation is controlled by the parameter :math:`\sigma_{max}` and its
    smoothness in the color space by :math:`k_{max}`. T

    You can find the formal definition in:
    `PRIME: A Few Primitives Can Boost Robustness to Common Corruptions <https://arxiv.org/abs/2112.13547>`.


    Args:
        k_max (int): Highest frequency component that defines the smoothness of the transformation.
        sigma_max (float): Maximum strength of the transformation.
        delta_bandwidth (float): Number of consecutive frequency components that define the transformation.
            If None, the bandwidth is set to k_max.
        sigma_min (float): Minimum strength of the transformation.
        same_on_batch (bool): Apply the same transformation across the batch.
        p (float): Probability of applying the transformation.
        keepdim (bool): Determines whether the output tensor has the same size as the input tensor.

    Returns:
        Tensor: Color jittered tensor image.

    Shape:
        - Input: :math:`(C, H, W)` or :math:`(B, C, H, W)`,
        - Output: :math:`(B, C, H, W)`

    .. note::
        This function internally uses :
        func:`MaxEntropyColorJitterGenerator`
    """

    def __init__(
        self,
        k_max: int,
        sigma_max: float = 0.01,
        delta_bandwidth: Optional[float] = None,
        sigma_min: float = 0.0,
        p: float = 1.0,
        p_batch: float = 1.0,
        same_on_batch: bool = False,
        keepdim: bool = False,
    ) -> None:
        super().__init__(
            p=p, p_batch=p_batch, same_on_batch=same_on_batch, keepdim=keepdim
        )

        self.k_max = k_max
        self.sigma_max = sigma_max
        self.delta_bandwidth = delta_bandwidth
        self.sigma_min = sigma_min

        self._param_generator = MaxEntropyColorJitterGenerator(
            k_max, sigma_max, delta_bandwidth, sigma_min
        )

    def apply_transform(
        self,
        input: Tensor,
        params: Dict[str, Tensor],
        flags: Dict[str, Any],
        transform: Optional[Tensor] = None,
    ) -> Tensor:
        r"""
        Applies the MaxEntropyColorJitter transformation to a given input tensor.

        Args:
            input (torch.Tensor): Input tensor to be transformed.
            params (Dict[str, torch.Tensor]): Dictionary of transformation parameters, including:
                * rand_coeffs (torch.Tensor): Tensor of random coefficients for each frequency component.
                * k (torch.Tensor): Tensor of frequency values for each color channel.
            flags (Dict[str, Any]): Dictionary of boolean flags for each transformation option.
            transform (Optional[torch.Tensor]): Not used in this transformation.

        Returns:
            torch.Tensor: Transformed tensor with a shape of :math:`(B, C, H, W)`.
        """
        # Reshape input to apply the transform
        input_shape = input.shape
        if len(input_shape) < 4:
            input = input[None, :]
            input_shape = input.shape

        colors = input.reshape(
            input_shape[0], input_shape[1], input_shape[2] * input_shape[3]
        )
        # Apply the transform
        freqs = torch.sin(
            colors[..., None] * params["k"][None, None, None, :] * torch.pi
        )
        transformed_colors = (
            torch.einsum("bcf,bcnf->bcn", params["rand_coeffs"], freqs) + colors
        )
        # Format the output
        transformed_colors = torch.clamp(transformed_colors, 0, 1)
        transformed_colors = transformed_colors.reshape(
            input_shape[0], input_shape[1], input_shape[2], input_shape[3]
        )

        return transformed_colors
