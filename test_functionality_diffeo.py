import imagenet_stubs
import matplotlib.pyplot as plt
import PIL.Image
import torch
import torchvision.transforms as T
from PRIME.augmentations.max_entropy_diffeomorphism import MaxEntropyDiffeomorphism

random_color = MaxEntropyDiffeomorphism(
    iT=1.0, jT=1.0, i_max=10.0, j_max=10.0, k_min=2, k_max=5, sigma_max=2.0, p_batch=0.5
)

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
    t=0 #plt.savefig("fig" + str(i) + ".png")
t = 0
