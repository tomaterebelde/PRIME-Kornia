import torch
from typing import Any, Dict, List, Optional
from kornia.core import Tensor

from PRIME.augmentations.max_entropy_color_jitter import MaxEntropyColorJitter
from PRIME.augmentations.max_entropy_random_filter import MaxEntropyRandomFilter
from PRIME.augmentations.max_entropy_diffeomorphism import MaxEntropyDiffeomorphism
from PRIME.random_generator.prime_generator import GeneralizedPRIMEModuleGenerator

from kornia.augmentation._2d.base import AugmentationBase2D


class PRIMEAugModule_Kornia(AugmentationBase2D):
    r"""
    Class to mix PRIME augmentations and to apply them to an input image.

    This transform follows the theroetical framework of PRIME,
    maximizing the entropy of the output image.
    To read more about the theory behind this transformation, please refer to the paper
    `PRIME: A Few Primitives Can Boost Robustness to Common Corruptions <https://arxiv.org/abs/2112.13547>`.

    Args:
        preprocess (Callable): Preprocessing function to be applied to the input image.
        p (float): Probability of applying the augmentation to the input image.
        p_batch (float): Probability of applying the augmentation to a batch of images.
        augmentations (List): List of augmentations to be applied to the input image. Defaults to the PRIME augmentations.
        mixture_width (int): Width of the mixture of transforms. Defaults to 3.
        mixture_depth (int): Depth of the mixture of transforms. If negative, the maximum depth is used. Defaults to -1.
        max_depth (int): Maximum depth of the mixture of transforms. Defaults to 3.
        same_on_batch (bool): Whether to apply the same augmentation to a batch of images. Defaults to False.
        Note that for PRIMEAugModule the same_on_batch only guarantees that the same mixture of transforms is applied to the batch.
        keepdim (bool): Whether to keep the input tensor dimensions the same. Defaults to False.

    Returns:
        torch.Tensor: Augmented image.

    Shape:
        - Input: :math:`(B, C, H, W)` or :math:`(C, H, W)`
        - Output: :math:`(B, C, H, W)`

    Examples: -

    .. note::
         This function internally uses :
           ' MaxEntropyColorJitter', 'MaxEntropyRandomFilter', 'MaxEntropyDiffeomorphism', 'PRIMEModuleGenerator'
    """

    def __init__(
        self,
        preprocess,
        augmentations: List = [
            MaxEntropyColorJitter(k_max=5, sigma_max=0.005, same_on_batch=True),
            MaxEntropyRandomFilter(kernel_size=3, same_on_batch=True),
            MaxEntropyDiffeomorphism(
                iT=1.0,
                jT=1.0,
                i_max=10.0,
                j_max=10.0,
                k_min=2,
                k_max=5,
                sigma_max=2.0,
                same_on_batch=True,
            ),
        ],
        mixture_width=3,
        mixture_depth=-1,
        max_depth=3,
        p: float = 1.0,  # TODO : should allways be 1, no? No.
        p_batch: float = 1.0,  ##TODO : should allways be 1, no? No.
        same_on_batch: bool = False,  # §TODO : Finish implementation of same_on_batch
        keepdim: bool = False,  # §TODO : Implementation of keepdim
    ) -> None:
        super().__init__(
            p=p, p_batch=p_batch, same_on_batch=same_on_batch, keepdim=keepdim
        )
        self.preprocess = preprocess
        self.augmentations = augmentations
        self.mixture_width = mixture_width
        self.max_depth = max_depth
        self.num_transforms = len(self.augmentations)
        self.depth = mixture_depth if mixture_depth > 0 else self.max_depth

        self.depth_combos = torch.tril(torch.ones((max_depth, max_depth)))

        self._param_generator = GeneralizedPRIMEModuleGenerator(
            depth_combos=self.depth_combos,
            num_transforms=self.num_transforms,
            mixture_width=mixture_width,
            depth=self.depth,
        )

    def transform_mask(self, x, mask_t):
        """
        Applies the augmentations in the augmentations list to x based on the mask.

        Args:
            x (Tensor): The input image tensor.
            mask_t (Tensor): A binary tensor representing the transformations to apply.

        Returns:
            aug_x (Tensor): The transformed image tensor.
        """
        aug_x = torch.zeros_like(x)
        for i in range(self.num_transforms):
            aug_x += self.augmentations[i](x) * mask_t[:, i]
        return aug_x

    def apply_transform(
        self,
        input: Tensor,
        params: Dict[str, Tensor],
        flags: Dict[str, Any],
        transform: Optional[Tensor] = None,
    ):
        # Fetch the parameters generated by the PRIMEModuleGenerator
        ws = params["ws"]
        m = params["m"]
        depth_idx = params["depth_idx"]
        trans_idx = params["trans_idx"]

        # Format the input tensor to match the mixture width of transformations
        img_repeat = input.repeat(self.mixture_width, 1, 1, 1, 1)
        img_repeat = img_repeat.reshape(
            self.mixture_width * input.shape[0],
            input.shape[1],
            input.shape[2],
            input.shape[3],
        )

        # Create the masks for the transformations
        trans_combos = torch.eye(self.num_transforms, device=self.device)
        depth_mask = torch.zeros(
            img_repeat.shape[0], self.max_depth, 1, 1, 1, device=self.device
        )
        trans_mask = torch.zeros(
            img_repeat.shape[0], self.num_transforms, 1, 1, 1, device=self.device
        )
        depth_mask.data[:, :, 0, 0, 0] = self.depth_combos[depth_idx]

        image_aug = img_repeat.clone()

        # Apply the transformations to the input image
        for d in range(self.depth):
            trans_mask.data[:, :, 0, 0, 0] = trans_combos[trans_idx[d]]
            image_aug.data = (
                depth_mask[:, d] * self.transform_mask(image_aug, trans_mask)
                + (1.0 - depth_mask[:, d]) * image_aug
            )

        # Reformat the output image to match the input shape
        if self.preprocess is not None:
            image_aug = self.preprocess(image_aug)

        image_aug = image_aug.reshape(
            self.mixture_width,
            input.shape[0],
            input.shape[1],
            input.shape[2],
            input.shape[3],
        )
        mix = torch.einsum("bm, mbchw -> bchw", ws, image_aug)
        if self.preprocess is not None:
            mixed = (1.0 - m) * self.preprocess(input) + m * mix
        else:
            mixed = (1.0 - m) * input + m * mix

        return mixed
