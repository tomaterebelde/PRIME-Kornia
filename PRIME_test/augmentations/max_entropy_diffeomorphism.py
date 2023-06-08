from typing import Any, Dict, Optional
import torch
import math
from opt_einsum import contract

from kornia.augmentation._2d.base import AugmentationBase2D
from kornia.core import Tensor
from PRIME.random_generator.max_entropy_diffeomorphism_generator import (
    MaxEntropyDiffeomorphismGenerator,
)

from kornia.geometry.transform import remap


class MaxEntropyDiffeomorphism(AugmentationBase2D):
    r"""
    Applies a random spatial transformation to a tensor image. This transform follows
    the theroetical framework of PRIME, maximizing the entropy of the output image.
    To read more about the theory behind this transformation, please refer to the paper
    `PRIME: A Few Primitives Can Boost Robustness to Common Corruptions <https://arxiv.org/abs/2112.13547>`.

    Args:
        iT (float): The number of control points along the y-axis for the grid of the displacement field.
        jT (float): The number of control points along the x-axis for the grid of the displacement field.
        i_max (float): The maximum displacement along the y-axis for each control point.
        j_max (float): The maximum displacement along the x-axis for each control point.
        k_min (float): The minimum value of the smoothing kernel standard deviation.
        k_max (float): The maximum value of the smoothing kernel standard deviation.
        sigma_max (float): The maximum value of the standard deviation of the random displacement field.
        p (float): Probability of applying the transformation. Default: 1.0.
        p_batch (float): Probability of applying the transformation to a batch of tensors. Default: 1.0.
        interpolation (str): Interpolation mode to calculate output values. Default: 'bilinear'.

    Returns:
        The randomly spatialy transformed input tensor.

    Shape:
        - Input: math:`(C, H, W)` or :math:`(B, C, H, W)`
        - Output: :math:`(B, C, H, W)`

    Examples: -

    .. note::
        This function internally uses :
            func:`MaxEntropyDiffeomorphismGenerator`

    """

    def __init__(
        self,
        iT,
        jT,
        i_max,
        j_max,
        k_min,
        k_max,
        sigma_max,
        interpolation="bilinear",
        p=1.0,
        p_batch=1.0,
        same_on_batch=False,
        keepdim=False,
    ):
        super().__init__(p=p, p_batch=p_batch, same_on_batch=same_on_batch, keepdim=keepdim)

        self.iT = iT
        self.jT = jT
        self.i_max = i_max
        self.j_max = j_max
        self.k_min = k_min
        self.k_max = k_max
        self.sigma_max = sigma_max
        self.flags = {"interpolation": interpolation}

        self._param_generator = MaxEntropyDiffeomorphismGenerator(
            iT, jT, i_max, j_max, k_min, k_max, sigma_max
        )



    def apply_transform(
        self,
        input: Tensor,
        params: Dict[str, Tensor],
        flags: Dict[str, Any],
        transform: Optional[Tensor] = None,
    ) -> Tensor:
        # Reshape input to apply the transform
        input_shape = input.shape
        if len(input_shape) < 4:
            input = input[None, :]

        # Get the scalar field to remap the input
        field_x, field_y = self._get_diffeomorphism_scalar_field(input, params)

        # Apply the transformation
        return remap(
            input,
            field_x,
            field_y,
            mode=flags["interpolation"],
            padding_mode="border",
            normalized_coordinates=True,
            align_corners=False,
        )    
    
    def _get_diffeomorphism_scalar_field(self, input: Tensor, params: Dict):
        r"""
        Generates a Random Diffeomorphism vector field given the input tensor's size.

        Args:
            input (torch.Tensor): A square input tensor of shape [B, C, H, W].
            params (Dict): Dictionary containing the parameters for the scalar field generation.
                'T' (float): Temperature for sampling displacement field.
                'cut' (int): High frequency cutoff for sampling the displacement field.

        Returns:
            xn (torch.Tensor): x-coordinate of the generated diffeomorphism field.
            yn (torch.Tensor): y-coordinate of the generated diffeomorphism field.

        """

        T = params["T"]
        cut = int(params["cut"])
        coeffs = params["coeffs"]

        n, m = input.shape[-2:]
        assert input.shape[-2] == n, "input(s) should be square."

        device = input.device

        # Sample dx, dy
        # u, v are defined in [0, 1]^2
        # dx, dx are defined in [0, n]^2
        u = self._scalar_field(n, cut, coeffs, device)  # [b, n, n]
        v = self._scalar_field(n, cut, coeffs, device)  # [b, n, n]
        dx = T**0.5 * u * n
        dy = T**0.5 * v * n

        # Generate the grid
        y, x = torch.meshgrid(
            torch.arange(n, dtype=dx.dtype, device=input.device),
            torch.arange(m, dtype=dx.dtype, device=input.device),
        )

        # Apply the displacement field
        xn = (x - dx).clamp(0, m - 1)
        yn = (y - dy).clamp(0, n - 1)

        # Normalize to [-1, 1]
        yn = yn / (n - 1) * 2 - 1
        xn = xn / (m - 1) * 2 - 1

        return xn, yn


    def _scalar_field(self, n, cut, coeffs, device="cpu"):
        """
        Generates a random scalar field of size nxn made of the first m modes.

        Args:
                b (int): Batch size.
                n (int): Size of the field.
                m (int): High frequency cutoff.

        Returns:
            (torch.Tensor): Random scalar field of size nxn.
        """
        e, s = self._scalar_field_modes(n, cut, dtype=torch.get_default_dtype(), device=device)
        c = coeffs * e

        #return contract("bij,xi,yj->byx", c, s, s)
        return torch.einsum('bij,xi,yj->byx', c, s, s)


    def _scalar_field_modes(self, n, cut, dtype=torch.float64, device="cpu"):
        """
        Generate the first m modes of a scalar field of size nxn.

        Args:
            n (int): Size of the field.
            m (int): High frequency cutoff.

        Returns:
            (torch.Tensor, torch.Tensor): sqrt(1 / Energy per mode) and the modes.
        """
        x = torch.linspace(0, 1, n, dtype=dtype, device=device)
        k = torch.arange(1, cut + 1, dtype=dtype, device=device)
        i, j = torch.meshgrid(k, k)
        r = (i.pow(2) + j.pow(2)).sqrt()
        e = (r < cut + 0.5) / r
        s = torch.sin(math.pi * x[:, None] * k[None, :])
        return e, s
