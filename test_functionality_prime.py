import torch
import imagenet_stubs
import matplotlib.pyplot as plt
import PIL.Image
import torchvision.transforms as T
from PRIME.augmentations.prime import PRIMEAugModule_Kornia
from transformation_layer import TransformLayer


batch = []
im = PIL.Image.open(imagenet_stubs.get_image_paths()[0])
for image_path in imagenet_stubs.get_image_paths():
    x = T.ToTensor()(im)
    batch.append(x)

batch = torch.stack(batch)
batch2 = batch.clone()

mean = torch.tensor([0.485, 0.456, 0.406])
std = torch.tensor([0.229, 0.224, 0.225])


prime_module = PRIMEAugModule_Kornia(
    preprocess=None,#TransformLayer(apply=False, mean=mean, std=std),
    mixture_width=3,
    mixture_depth=-1,
    max_depth=3,
    p=1,
    p_batch=1.,
    same_on_batch=False,
    keepdim=False,
)

batch2 = prime_module(batch2)
for i in range(batch2.shape[0]):
    plt.subplot(1, 2, 1)
    plt.imshow(batch[i].permute(1, 2, 0))
    plt.subplot(1, 2, 2)
    plt.imshow(batch2[i].permute(1, 2, 0))
    t = 0
t = 0
