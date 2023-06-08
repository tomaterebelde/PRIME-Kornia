from typing import Any, Dict, Optional
from PRIME.random_generator.max_entropy_rand_filter_generator import (
    MaxEntropyRandomFilterGenerator,
)

import torch
from kornia.augmentation._2d.intensity.base import IntensityAugmentationBase2D
from kornia.core import Tensor


class MaxEntropyRandomFilter(IntensityAugmentationBase2D):
    r"""
    Applies a random filtering transformation to a tensor image. This transform follows
    the theroetical framework of PRIME, maximizing the entropy of the output image.
    To read more about the theory behind this transformation, please refer to the paper
    `PRIME: A Few Primitives Can Boost Robustness to Common Corruptions <https://arxiv.org/abs/2112.13547>`.


    Args:
        kernel_size (int): size of the kernel to convolve the input tensor with.
        sigma_max (float): maximum value of the standard deviation to use on the transformation strength.
        sigma_min (float): minimum value of the standard deviation to use on the transformation strength.
        same_on_batch (bool): apply the same transformation across the batch.
        p (float): probability of applying the transformation.
        keepdim (bool): determines whether the output tensor has the same size as the input tensor.

    Returns:
        Tensor: Filtered tensor image.

    Shape:
        - Input: math:`(C, H, W)` or :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H, W)`

    Examples: -
    Input: tensor([[[0.9663, 0.9385, 0.0160],
                    [0.3854, 0.6294, 0.0780],
                    [0.8543, 0.1722, 0.4978]]])

    Output: tensor([[[[1.0000, 1.0000, 0.5302],
                    [1.0000, 1.0000, 0.0000],
                    [1.0000, 1.0000, 0.0000]]]])

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
        super().__init__(p=p, p_batch=p_batch, same_on_batch=same_on_batch, keepdim=keepdim)

        self.kernel_size = kernel_size
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min

        self._param_generator = MaxEntropyRandomFilterGenerator(
            kernel_size, sigma_max, sigma_min
        )  # This will generate the parameters for the augmentation

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
