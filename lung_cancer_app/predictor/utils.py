from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import os
import copy
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.models import resnet50, ResNet50_Weights
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

# Model architecture
def create_resnet_model(num_classes, use_pretrained=True):
    if use_pretrained:
        weights = ResNet50_Weights.IMAGENET1K_V1
    else:
        weights = None
    model = resnet50(weights=weights)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, num_classes)
    )
    return model


def load_model(model_path, num_classes, device='cpu'):
    model = create_resnet_model(num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def preprocess_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    return image

#predictions
def predict_image(model, image_tensor, class_names, device='cpu'):
    with torch.no_grad():
        outputs = model(image_tensor.to(device))
        probs = torch.softmax(outputs, dim=1)
        predicted_class_idx = probs.argmax().item()
        predicted_class_name = class_names[predicted_class_idx]
        return predicted_class_name, probs.squeeze().tolist()