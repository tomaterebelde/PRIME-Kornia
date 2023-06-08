import numpy as np
import torch
from einops import parse_shape, rearrange


class RandomFilter(torch.nn.Module):
    def __init__(self, kernel_size, sigma, stochastic=False, sigma_min=0.):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma

        self.stochastic = stochastic
        if self.stochastic:
            self.kernels_size_candidates = torch.tensor([float(i) for i in range(self.kernel_size, self.kernel_size + 2, 2)])
            self.sigma_min = sigma_min
            self.sigma_max = sigma

    def forward(self, img):
        if self.stochastic:
            self._sample_params()

        init_shape = img.shape
        if len(init_shape) < 4:
            img = rearrange(img, "c h w -> () c h w")

        shape_dict = parse_shape(img, "b c h w")
        batch_size = shape_dict["b"]
        img = rearrange(img, "b c h w -> c b h w")

        delta = torch.zeros((1, self.kernel_size, self.kernel_size), device=img.device)
        center = int(np.ceil(self.kernel_size / 2))
        delta[0, center, center] = 1.0

        conv_weight = rearrange(
            self.sigma * torch.randn((batch_size, self.kernel_size, self.kernel_size), device=img.device) + delta,
            "b h w -> b (h w)",
        )

        return self.apply_transform(img, {"conv_weight": conv_weight})
    
    def apply_transform(self, input, params):
        input_shape = input.shape
        if len(input_shape) < 4:
            input = input[None, :]
            input_shape = input.shape
        input_shape = parse_shape(input, "b c h w")

        conv_weight = params["conv_weight"]

        input = rearrange(input, "b c h w -> c b h w")

        filtered_img = torch.nn.functional.conv2d(
            input, conv_weight, padding="same", groups=input_shape["b"]
        )
        

        # Deal with NaN values due to mixed precision -> Convert them to 1.
        filtered_img[filtered_img.isnan()] = 1.

        filtered_img = rearrange(filtered_img, "c b h w -> b c h w")
        filtered_img = torch.clamp(filtered_img, 0., 1.).reshape(input_shape["b"], input_shape["c"], input_shape["h"], input_shape["w"])

        return filtered_img

    def _sample_params(self):
        self.kernel_size = int(self.kernels_size_candidates[torch.multinomial(self.kernels_size_candidates, 1)].item())
        self.sigma = torch.FloatTensor([1]).uniform_(self.sigma_min, self.sigma_max).item()

    def __repr__(self):
        return self.__class__.__name__ + f"(sigma={self.sigma}, kernel_size={self.kernel_size})"


if __name__ == "__main__":
    import imagenet_stubs
    import matplotlib.pyplot as plt
    import PIL.Image
    import torchvision.transforms as T

    random_filter = RandomFilter(kernel_size=3, sigma=0.1, stochastic=True, sigma_min=0.01)
    
    batch = []
    for image_path in imagenet_stubs.get_image_paths():
        im = PIL.Image.open(image_path)
        x = T.ToTensor()(im)
        batch.append(x)

    batch = torch.stack(batch)
    batch2 = random_filter(batch)

    plt.subplot(1,2,1)
    plt.imshow(batch[0].permute(1, 2, 0))
    plt.subplot(1,2,2)
    plt.imshow(batch2[0].permute(1, 2, 0))
    t = 0
