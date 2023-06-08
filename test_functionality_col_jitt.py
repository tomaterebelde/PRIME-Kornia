import imagenet_stubs
import matplotlib.pyplot as plt
import PIL.Image
import torch
import torchvision.transforms as T
from PRIME.augmentations.max_entropy_color_jitter import MaxEntropyColorJitter

random_color = MaxEntropyColorJitter(k_max=100, sigma_max=0.02, keepdim=True)


batch = []
im = PIL.Image.open(imagenet_stubs.get_image_paths()[0])
for image_path in imagenet_stubs.get_image_paths():
    x = T.ToTensor()(im)
    batch.append(x)

batch = torch.stack(batch)
batch2 = random_color(batch)
for i in range(batch.shape[0]):
    plt.subplot(1, 2, 1)
    plt.imshow(batch[i].permute(1, 2, 0))
    plt.subplot(1, 2, 2)
    plt.imshow(batch2[i].permute(1, 2, 0))
    plt.savefig("fig" + str(i) + ".png")
t = 0
