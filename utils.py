from PIL import Image
import torchvision.transforms as transforms
import torch

def load_image(image_path, max_size=512):
    image = Image.open(image_path).convert('RGB')
    size = min(max(image.size), max_size)
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x[:3, :, :]),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def save_image(tensor, path):
    image = tensor.clone().squeeze(0)
    image = image.cpu().detach()
    unnormalize = transforms.Normalize(
        mean=[-2.118, -2.036, -1.804],
        std=[4.367, 4.464, 4.444]
    )
    image = unnormalize(image)
    image = transforms.ToPILImage()(image.clamp(0, 1))
    image.save(path)

import matplotlib.pyplot as plt

def imshow(tensor, title=None):
    # Disabled to avoid freezing due to GUI issues or plt.show()
    # image = tensor.clone().squeeze(0).cpu()
    # unnormalize = transforms.Normalize(
    #     mean=[-2.118, -2.036, -1.804],
    #     std=[4.367, 4.464, 4.444]
    # )
    # image = unnormalize(image)
    # image = image.clamp(0, 1)
    # image = transforms.ToPILImage()(image)

    # plt.imshow(image)
    # if title:
    #     plt.title(title)
    # plt.axis('off')
    pass
