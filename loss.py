import torch
import torch.nn.functional as F

def gram_matrix(tensor):
    _, c, h, w = tensor.size()
    tensor = tensor.view(c, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram / (c * h * w)

def content_loss(target_features, content_features):
    return F.mse_loss(target_features, content_features)

def style_loss(target_features, style_grams):
    loss = 0
    for layer in style_grams:
        target_gram = gram_matrix(target_features[layer])
        style_gram = style_grams[layer]
        loss += F.mse_loss(target_gram, style_gram)
    return loss
