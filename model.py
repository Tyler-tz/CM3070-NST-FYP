import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import VGG19_Weights

def get_model_and_layers():
    weights = VGG19_Weights.DEFAULT
    vgg = models.vgg19(weights=weights).features.eval()

    for param in vgg.parameters():
        param.requires_grad = False

    # Map layer names manually to match what's used in NST
    model = nn.Sequential()
    content_layers = []
    style_layers = []

    conv_idx = 0
    relu_idx = 0
    pool_idx = 0
    layer_names = []

    for i, layer in enumerate(vgg.children()):
        if isinstance(layer, nn.Conv2d):
            conv_idx += 1
            name = f"conv_{conv_idx}"
        elif isinstance(layer, nn.ReLU):
            name = f"relu_{conv_idx}"
            layer = nn.ReLU(inplace=False)  # Needed for backward
        elif isinstance(layer, nn.MaxPool2d):
            pool_idx += 1
            name = f"pool_{pool_idx}"
        elif isinstance(layer, nn.BatchNorm2d):
            name = f"bn_{conv_idx}"
        else:
            name = f"unknown_{i}"

        model.add_module(name, layer)
        layer_names.append(name)

        if name == 'conv_4':
            content_layers = [name]
        if name in ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']:
            style_layers.append(name)

    return model, content_layers, style_layers
