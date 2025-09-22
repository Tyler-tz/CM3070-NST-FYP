import torch
from torchvision import transforms
from PIL import Image
from utils import load_image, imshow, save_image
from model import get_model_and_layers
from style_transfer import run_style_transfer

# Configuration
content_path = "images/content1.jpg"
style_path = "images/style.jpg"
output_path = "images/output_content1.2.jpg"
image_size = 512
num_steps = 300
content_weight = 1
style_weight = 5e5

# Load and preprocess images
content_img = load_image(content_path, image_size)
style_img = load_image(style_path, image_size)

# Show input images
imshow(content_img, title='Content Image')
imshow(style_img, title='Style Image')

# Load model
vgg, content_layers, style_layers = get_model_and_layers()

# Generate input image (copy of content image)
input_img = content_img.clone()

# Run style transfer
output = run_style_transfer(vgg, content_img, style_img, input_img,
                            content_layers, style_layers,
                            content_weight=content_weight,
                            style_weight=style_weight,
                            num_steps=num_steps)

# Show and save result
# imshow(output, title='Output Image')
save_image(output, output_path)

# Display all figures
import matplotlib.pyplot as plt
plt.show()