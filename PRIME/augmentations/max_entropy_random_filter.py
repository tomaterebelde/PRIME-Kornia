from typing import Any, Dict, Optional
from PRIME.random_generator.max_entropy_random_filter_generator import (
    MaxEntropyRandomFilterGenerator,
)

import torch
from kornia.augmentation._2d.intensity.base import IntensityAugmentationBase2D
from kornia.core import Tensor


class MaxEntropyRandomFilter(IntensityAugmentationBase2D):
    r"""Apply a maximum entropy spectral transformation to a tensor image.

    This is one of the three base transformations that defines PRIME. The strength of
    the transformation is controlled by the parameter :math:`\sigma_{max}` and its
    smoothness in the frequency domain by :math:`kernel_size`.

    You can find the formal definition in:
    `PRIME: A Few Primitives Can Boost Robustness to Common Corruptions <https://arxiv.org/abs/2112.13547>`.

    Args:
        kernel_size (int): Size of the kernel to convolve the input tensor with. It controls the smoothness
            of the transformation.
        sigma_max (float): Maximum strength of the transformation.
        sigma_min (float): Minimum strength of the transformation.
        same_on_batch (bool): Apply the same transformation across the batch.
        p (float): Probability of applying the transformation.
        keepdim (bool): Determines whether the output tensor has the same size as the input tensor.

    Returns:
        The randomly filtered input tensor.

    Shape:
        - Input: math:`(C, H, W)` or :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H, W)`

    .. note::
        This function internally uses :
            func:`MaxEntropyRandomFilterGenerator`
    """

    def __init__(
        self,
        kernel_size,
        sigma_max=0.4,
        sigma_min=0.0,
        p: float = 1.0,
        p_batch: float = 1.0,
        same_on_batch: bool = False,
        keepdim: bool = False,
    ) -> None:
        super().__init__(
            p=p, p_batch=p_batch, same_on_batch=same_on_batch, keepdim=keepdim
        )

        self.kernel_size = kernel_size
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min

        self._param_generator = MaxEntropyRandomFilterGenerator(
            kernel_size, sigma_max, sigma_min
        )

    def apply_transform(
        self,
        input: Tensor,
        params: Dict[str, Tensor],
        flags: Dict[str, Any],
        transform: Optional[Tensor] = None,
    ) -> Tensor:
        r"""
        Applies the Max Entropy Filter to the input tensor.

        Args:
            input (torch.Tensor): Input tensor to be transformed.
            params (Dict[str, Tensor]): A dictionary containing the following parameter:
                - conv_weight (Tensor): The convolutional weight of the Max Entropy filter.
            flags (Dict[str, Any]): A dictionary containing the following optional flags:
                - `margins`: number of pixels to be padded to the sides of the image.
            transform (Optional[Tensor]): Not used.

        Returns:
            torch.Tensor: Transformed tensor with the same shape as the input tensor.
        """

        # Reshape input to apply the transform
        input_shape = input.shape
        if len(input_shape) < 4:
            input = input[None, :]
            input_shape = input.shape

        img = torch.permute(input, (1, 0, 2, 3))
        # Apply the transform
        filtered_img = torch.nn.functional.conv2d(
            img, params["conv_weight"], padding="same", groups=input_shape[0]
        )
        # Deal with NaN values due to mixed precision -> Convert them to 1.
        filtered_img[filtered_img.isnan()] = 1.0
        filtered_img = torch.permute(filtered_img, (1, 0, 2, 3))
        filtered_img = torch.clamp(filtered_img, 0.0, 1.0)

        return filtered_img
