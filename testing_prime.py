import os
import unittest

import torch
from functorch import make_functional_with_buffers

from PRIME_test.augmentations.prime import PRIMEAugModule_Kornia
from PRIME_test.random_generator.prime_generator import GeneralizedPRIMEModuleGenerator
from old_prime.prime import GeneralizedPRIMEModule, PRIMEAugModule
from kornia_implementation.transformation_layer import TransformLayer
import PIL.Image
import imagenet_stubs
import matplotlib.pyplot as plt


import torchvision.transforms as T

class mock_augmentation(torch.nn.Module):
    def __init__(self, div=2):
        super().__init__()
        self.div = div
        
    def forward(self, input):
        input_shape = input.shape
        if len(input_shape) < 4:
            input = input[None, :]
            input_shape = input.shape
        return input/self.div


class PRIMETest(unittest.TestCase):
    """Tests the implementation of Prime."""


    def test_apply_transform(self):
        """Tests the linearization of a model."""
        # Init a tensor
        image = PIL.Image.open(imagenet_stubs.get_image_paths()[0])
        image = T.ToTensor()(image)
        image = image[None, :]
        a = [ mock_augmentation(div=2), mock_augmentation(div=3), mock_augmentation(div=4)]
        augment = PRIMEAugModule(augmentations=a)
       
        depth_combos = torch.tril(torch.ones((3,3)))

        # Get random parameters
        parameters = GeneralizedPRIMEModuleGenerator(depth_combos = depth_combos,
        num_transforms=3,
        mixture_width=3,
        depth=3,)
        parameters = parameters(image.shape, same_on_batch=False)

        # Initiallize the transforms
        Prime_aug = PRIMEAugModule_Kornia( preprocess=None,
                                          augmentations = a,
            mixture_width=3,
            mixture_depth=-1,
            max_depth=3,
            p=1,
            p_batch=1.,
            same_on_batch=False,
            keepdim=False,)
        aug =GeneralizedPRIMEModule(preprocess=TransformLayer(apply=False, mean = 0, std =0),
                                    aug_module=augment, )

        # Apply the transforms
        image1 = Prime_aug.apply_transform(image, parameters, flags=None, transform=None)
        image2 = aug.apply_t(image, parameters)

        image1 = torch.squeeze(image1)
        image2 = torch.squeeze(image2)

        mse = torch.nn.MSELoss()
        error = mse(image1, image2)
        print("Error: ", error)

        plt.subplot(1,2,1)
        plt.imshow(image1.permute(1, 2, 0))
        plt.subplot(1,2,2)
        plt.imshow(image2.permute(1, 2, 0))
        t=0

        assert torch.allclose(image1, image2)
    
    def test_keep_dim_true(self):
        # Init tensor
        tensor = torch.rand(3, 24, 24)
        tensor_shape1 = tensor.shape

        # Init the transforms
        Prime_aug = PRIMEAugModule_Kornia(
                preprocess=None,
                mixture_width=3,
                mixture_depth=-1,
                max_depth=3,
                p=1,
                p_batch=1.,
                same_on_batch=False,
                keepdim=True,)
        tensor2 = Prime_aug(tensor)
        tensor_shape2 = tensor2.shape

        # Check if the shape is the same
        assert tensor_shape1 == tensor_shape2

    def test_keep_dim_false(self):
        # Init tensor
        tensor = torch.rand(3, 24, 24)
        print(tensor)
        tensor_shape1 = tensor.shape

        # Init the transforms
        Prime_aug = PRIMEAugModule_Kornia(
                preprocess=None,
                mixture_width=3,
                mixture_depth=-1,
                max_depth=3,
                p=1,
                p_batch=1.,
                same_on_batch=False,
                keepdim=False,)
        tensor2 = Prime_aug(tensor)
        tensor_shape2 = tensor2.shape
        print(tensor2)

        # Check if the shape is the same
        assert tensor_shape1 != tensor_shape2



    def test_same_on_batch_true(self):
        batch = []
        im = PIL.Image.open(imagenet_stubs.get_image_paths()[0])
        for image_path in imagenet_stubs.get_image_paths():
            x = T.ToTensor()(im)
            batch.append(x)

        batch = torch.stack(batch)
        Prime_aug = PRIMEAugModule_Kornia(
                preprocess=None,
                mixture_width=3,
                mixture_depth=-1,
                max_depth=3,
                p=1,
                p_batch=1.,
                same_on_batch=True,
                keepdim=False,)
        batch2 = Prime_aug(batch)
        if torch.allclose(batch[0], batch2[0]):
            assert False
        for i in range(batch.shape[0]):
            if not torch.allclose(batch2[i], batch2[0]):
                assert False
        assert True


    def test_same_on_batch_false(self):
        batch = []
        im = PIL.Image.open(imagenet_stubs.get_image_paths()[0])
        for image_path in imagenet_stubs.get_image_paths():
            x = T.ToTensor()(im)
            batch.append(x)

        batch = torch.stack(batch)
        Prime_aug = PRIMEAugModule_Kornia(
                preprocess=None,
                mixture_width=3,
                mixture_depth=-1,
                max_depth=3,
                p=1,
                p_batch=1.,
                same_on_batch=False,
                keepdim=False,)
        batch2 = Prime_aug(batch)

        for i in range(batch.shape[0]-1):
            if torch.allclose(batch2[i+1], batch2[0]):
                assert False
        assert True

    

if __name__ == "__main__":
    unittest.main()