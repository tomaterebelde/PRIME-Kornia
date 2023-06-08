import torch
import torch.nn as nn
from typing import List
from einops import parse_shape, rearrange
import numpy as np
import torchvision.transforms as T

class ChannelEnhance(nn.Module):

    def __init__(self, enhance_coefs: List) -> None:
        """Enhances the contrast of each channel of a batch of images by a given coefficient."""
        super().__init__()
        self.enhance_coefs = enhance_coefs
    
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        ## imgs: (batch_size, channels, height, width)
        init_shape = img.shape
        if len(init_shape) < 4:
            img = rearrange(img, "c h w -> () c h w")
        
        return self.channel_enhancement(img, self.enhance_coefs)
        
    def channel_enhancement(self, img, enhance_coefs):

        img_shape = parse_shape(img, "b c h w")

        colors = rearrange(img, "b c h w -> b c (h w)")
        enhance_coef_tensor = colors **2 * self.enhance_coefs[0] + colors * self.enhance_coefs[1] + self.enhance_coefs[2]
        enhanced_images = colors * enhance_coef_tensor
        enhanced_images = rearrange(enhanced_images, "b c (h w) -> b c h w", ** img_shape)

        return torch.clamp(enhanced_images, 0, 1)

if __name__ == "__main__":
    import imagenet_stubs
    import matplotlib.pyplot as plt
    import PIL.Image
    
    batch = []
    for image_path in imagenet_stubs.get_image_paths():
        im = PIL.Image.open(image_path)
        x = T.ToTensor()(im)
        batch.append(x)

    enhance_array = [-1, 2, 1]
    module = ChannelEnhance(enhance_array)

    
    ## batch is a list before torch.stack  
    batch = torch.stack(batch)
    ## it is now a tensor of 16 images, 3 colors, 299 h , 299 w
    batch2 = module(batch)
    plt.subplot(1,2,1)
    plt.imshow(batch[9].permute(1, 2, 0))
    plt.subplot(1,2,2)
    plt.imshow(batch2[9].permute(1, 2, 0))
    plt.savefig("fig1.png")
    t = 0


