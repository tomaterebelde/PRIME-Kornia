from typing import Dict, List, Tuple

import torch
from torch.distributions import Uniform, Dirichlet, Beta

from kornia.augmentation.random_generator.base import RandomGeneratorBase
from kornia.augmentation.utils import _adapted_rsampling, _common_param_check
from kornia.core import Tensor
from kornia.utils.helpers import _extract_device_dtype


class GeneralizedPRIMEModuleGenerator(RandomGeneratorBase):
    r"""
    Class to generate the parameters for random PRIME mixing.

    This transform follows the theroetical framework of PRIME,
    maximizing the entropy of the output image.
    To read more about the theory behind this transformation, please refer to the paper
    `PRIME: A Few Primitives Can Boost Robustness to Common Corruptions <https://arxiv.org/abs/2112.13547>`.

    Args:
        depth_combos (Tensor): Tensor representing the combination of depth and transform indices to be used
            in generating the PRIME modules.
        num_transforms (int): The number of available transforms in each depth of the PRIME module.
        mixture_width (int): The width of the mixture of transforms.
        depth (int): The maximum depth of the mixture of transforms.

    Returns:
        Dict[str, Tensor]: a dictionary containing the weights 'ws', the mixing coefficients 'm', the depth indices 'depth_idx'
            and the transformation indices 'trans_idx'.
    """

    def __init__(
        self,
        depth_combos,
        num_transforms=3,
        mixture_width=3,
        depth=3,
    ) -> None:
        super().__init__()
        self.depth_combos = depth_combos
        self.mixture_width = mixture_width
        self.depth = depth
        self.num_transforms = num_transforms

    def __repr__(self) -> str:
        repr = f"depth_combos={self.depth_combos}, num_transforms={self.num_transforms}, mixture_width={self.mixture_width}, depth={self.depth}"
        return repr

    def make_samplers(self, device: torch.device, dtype: torch.dtype) -> None:
        self.dirichlet_sampler = Dirichlet(
            concentration=torch.tensor([1.0] * self.mixture_width, device=device)
        )
        self.beta_sampler = Beta(
            concentration1=torch.ones(1, device=device, dtype=torch.float32),
            concentration0=torch.ones(1, device=device, dtype=torch.float32),
        )
        self.depth_idx_sampler = Uniform(low=0, high=len(self.depth_combos))
        self.trans_idx_sampler = Uniform(low=0, high=self.num_transforms)

    def forward(
        self, batch_shape: Tuple[int, ...], same_on_batch: bool = False
    ) -> Dict[str, Tensor]:
        _common_param_check(batch_shape[0], same_on_batch)
        _device, _dtype = _extract_device_dtype(
            [self.mixture_width, self.depth, same_on_batch]
        )

        # Sample the parameters
        ws = _adapted_rsampling(
            (batch_shape[0],), self.dirichlet_sampler, same_on_batch = False
        )
        m = _adapted_rsampling((batch_shape[0],), self.beta_sampler, same_on_batch = False)[
            ..., None, None
        ]
        depth_idx = _adapted_rsampling(
            (batch_shape[0] * self.mixture_width,),
            self.depth_idx_sampler,
            same_on_batch,
        ).int()
        
        if same_on_batch:
            for i in range(batch_shape[0]):
                ws[i] = ws[0]
                m[i] = m[0]
            trans_idx = _adapted_rsampling((batch_shape[0]*self.mixture_width, ), self.trans_idx_sampler, same_on_batch).int()
            trans_idx = trans_idx.repeat(self.depth, 1)
        
        else :
            trans_idx = _adapted_rsampling(
                (
                    self.depth,
                    batch_shape[0] * self.mixture_width,
                ),
                self.trans_idx_sampler,
                same_on_batch,
            ).int()

        # Return generated parameters
        return dict(
            ws=ws.to(device=_device, dtype=_dtype),
            m=m.to(device=_device, dtype=_dtype),
            depth_idx=depth_idx.to(device=_device, dtype=torch.long),
            trans_idx=trans_idx.to(device=_device, dtype=torch.long),
        )