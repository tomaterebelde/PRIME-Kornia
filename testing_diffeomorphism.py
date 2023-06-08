import os
import unittest

import torch
from functorch import make_functional_with_buffers
from PRIME_test.augmentations.max_entropy_diffeomorphism import MaxEntropyDiffeomorphism
from PRIME_test.random_generator.max_entropy_diffeomorphism_generator import MaxEntropyDiffeomorphismGenerator
from old_prime.diffeomorphism import deform
import PIL.Image
import imagenet_stubs
import matplotlib.pyplot as plt


import torchvision.transforms as T


class DiffeomorphismTest(unittest.TestCase):
    """Tests the implementation of Color Jitter."""


    def test_apply_transform(self):
        """Tests the linearization of a model."""
        # Init a tensor
        image = PIL.Image.open(imagenet_stubs.get_image_paths()[0])
        image = T.ToTensor()(image)
        image = image[None, :]
       

        # Get random parameters
        parameters = MaxEntropyDiffeomorphismGenerator(iT=1.0, jT=1.0, i_max=10.0, j_max=10.0, k_min=2, k_max=5, sigma_max=2.0)
        parameters = parameters(image.shape, same_on_batch=False)

        # Initiallize the transforms
        Prime_diffeo = MaxEntropyDiffeomorphism(iT=1.0, jT=1.0, i_max=10.0, j_max=10.0, k_min=2, k_max=5, sigma_max=2.0)
        # diffeo = Diffeo(sT = 1., rT = 1., scut = 10., rcut = 10., cutmin = 2, cutmax = 5, alpha = 2.0)

        # Apply the transforms
        image1 = Prime_diffeo.apply_transform(image, parameters, flags={"interpolation": "bilinear"}, transform=None)
        image2 = deform(image, parameters)

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
        Prime_diffeo = MaxEntropyDiffeomorphism(iT=1.0, jT=1.0, i_max=10.0, j_max=10.0, k_min=2, k_max=5, sigma_max=2.0, keepdim=True)
        tensor2 = Prime_diffeo(tensor)
        tensor_shape2 = tensor2.shape

        # Check if the shape is the same
        assert tensor_shape1 == tensor_shape2

    def test_keep_dim_false(self):
        # Init tensor
        tensor = torch.rand(3, 24, 24)
        tensor_shape1 = tensor.shape

        # Init the transforms
        Prime_diffeo = MaxEntropyDiffeomorphism(iT=1.0, jT=1.0, i_max=10.0, j_max=10.0, k_min=2, k_max=5, sigma_max=2.0, keepdim=False)
        tensor2 = Prime_diffeo(tensor)
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
        Prime_diffeo = MaxEntropyDiffeomorphism(iT=1.0, jT=1.0, i_max=10.0, j_max=10.0, k_min=2, k_max=5, sigma_max=2.0, same_on_batch=True)
        batch2 = Prime_diffeo(batch)
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
        Prime_diffeo = MaxEntropyDiffeomorphism(iT=1.0, jT=1.0, i_max=10.0, j_max=10.0, k_min=2, k_max=5, sigma_max=2.0, same_on_batch=False)
        batch2 = Prime_diffeo(batch)

        for i in range(batch.shape[0]-1):
            if torch.allclose(batch2[i+1], batch2[0]):
                assert False
        assert True
    




if __name__ == "__main__":
    unittest.main()