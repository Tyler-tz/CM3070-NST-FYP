import torch
import torch.nn.functional as F

def gram_matrix(input):
    b, c, h, w = input.size()
    features = input.view(b * c, h * w)
    G = torch.mm(features, features.t())
    return G.div(b * c * h * w)

def get_features(x, model, layers):
    features = {}
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[name] = x
    return features

def run_style_transfer(model, content_img, style_img, input_img, content_layers, style_layers,
                       content_weight=1, style_weight=1e6, num_steps=300):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    input_img = input_img.to(device).requires_grad_(True)
    content_img = content_img.to(device)
    style_img = style_img.to(device)

    optimizer = torch.optim.LBFGS([input_img])

    # Precompute target features
    content_features = get_features(content_img, model, content_layers)
    style_features = get_features(style_img, model, style_layers)
    style_grams = {name: gram_matrix(style_features[name]) for name in style_features}

    run = [0]
    while run[0] <= num_steps:
        def closure():
            optimizer.zero_grad()
            target_features = get_features(input_img, model, content_layers + style_layers)

            # Content loss
            c_loss = 0
            for cl in content_layers:
                c_loss += F.mse_loss(target_features[cl], content_features[cl])

            # Style loss
            s_loss = 0
            for sl in style_layers:
                target_gram = gram_matrix(target_features[sl])
                style_gram = style_grams[sl]
                s_loss += F.mse_loss(target_gram, style_gram)

            total_loss = content_weight * c_loss + style_weight * s_loss
            total_loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print(f"Step {run[0]}/{num_steps}, Content Loss: {c_loss.item():.4f}, Style Loss: {s_loss.item():.4f}")

            return total_loss

        optimizer.step(closure)

    return input_img.detach()
