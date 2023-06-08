import os
import unittest

import torch
from functorch import make_functional_with_buffers
from PRIME_test.augmentations.max_entropy_random_filter import MaxEntropyRandomFilter
from PRIME_test.random_generator.max_entropy_rand_filter_generator import MaxEntropyRandomFilterGenerator
from old_prime.rand_filter import RandomFilter
import PIL.Image
import imagenet_stubs
import matplotlib.pyplot as plt


import torchvision.transforms as T


class RandomFilterTest(unittest.TestCase):
    """Tests the implementation of Color Jitter."""


    def test_apply_transform(self):
        """Tests the linearization of a model."""
        # Init a tensor
        image = PIL.Image.open(imagenet_stubs.get_image_paths()[0])
        image = T.ToTensor()(image)
        image = image[None, :]
       

        # Get random parameters
        parameters = MaxEntropyRandomFilterGenerator(kernel_size=3,sigma_max=0.9, sigma_min=0.75)
        parameters = parameters(image.shape, same_on_batch=False)

        # Initiallize the transforms
        Prime_rand_filter = MaxEntropyRandomFilter(kernel_size=3,sigma_max=0.9, sigma_min=0.75)
        rand_filter = RandomFilter(kernel_size=3, sigma=0.9, stochastic=True, sigma_min=0.75)

        # Apply the transforms
        image1 = Prime_rand_filter.apply_transform(image, parameters, flags=None, transform=None)
        image2 = rand_filter.apply_transform(image, parameters)

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
        Prime_rand_filter = MaxEntropyRandomFilter(kernel_size=3,sigma_max=0.9, sigma_min=0.75, keepdim=True)
        tensor2 = Prime_rand_filter(tensor)
        tensor_shape2 = tensor2.shape

        # Check if the shape is the same
        assert tensor_shape1 == tensor_shape2

    def test_keep_dim_false(self):
        # Init tensor
        tensor = torch.rand(3, 24, 24)
        tensor_shape1 = tensor.shape

        # Init the transforms
        Prime_rand_filter = MaxEntropyRandomFilter(kernel_size=3,sigma_max=0.9, sigma_min=0.75, keepdim=False)
        tensor2 = Prime_rand_filter(tensor)
        tensor_shape2 = tensor2.shape

        # Check if the shape is the same
        assert tensor_shape1 != tensor_shape2



    def test_same_on_batch_true(self):
        batch = []
        im = PIL.Image.open(imagenet_stubs.get_image_paths()[0])
        for image_path in imagenet_stubs.get_image_paths():
            x = T.ToTensor()(im)
            batch.append(x)

        batch = torch.stack(batch)
        Prime_rand_filter = MaxEntropyRandomFilter(kernel_size=3,sigma_max=0.9, sigma_min=0.75, same_on_batch=True)
        batch2 = Prime_rand_filter(batch)
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
        Prime_rand_filter = MaxEntropyRandomFilter(kernel_size=3,sigma_max=0.9, sigma_min=0.75, same_on_batch=False)
        batch2 = Prime_rand_filter(batch)

        for i in range(batch.shape[0]-1):
            if torch.allclose(batch2[i+1], batch2[0]):
                assert False
        assert True
    




if __name__ == "__main__":
    unittest.main()