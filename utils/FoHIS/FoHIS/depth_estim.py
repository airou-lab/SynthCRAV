import torch
import cv2
import numpy as np
from torchvision import transforms

def load_torch_model():
    # Load MiDaS model and set it to evaluation mode
    model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")

    # Set up the transform for the input image
    transform = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform

    return model, transform

def generate_depth_map(img,filename,model,transform):
    # Load the input image
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_tensor = transform(img_rgb).unsqueeze(0)

    # Perform inference
    with torch.no_grad():
        depth = model(input_tensor)

    # Post-process the output
    depth = depth.squeeze().cpu().numpy()
    depth = (depth - depth.min()) / (depth.max() - depth.min())  # Normalize to 0-1
    depth = (depth * 255).astype(np.uint8)  # Convert to 8-bit
    
    cv2.imwrite('./'+filename+'d')