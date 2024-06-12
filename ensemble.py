import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import numpy as np
import os
from efficientnet_pytorch import EfficientNet
import timm

# Define transformations for the validation dataset
data_transforms = {
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = 'dataset'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                  for x in ['val']}
dataloaders = {x: DataLoader(image_datasets[x], batch_size=32, shuffle=True, num_workers=4)
               for x in ['val']}
class_names = image_datasets['val'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load the pre-trained ResNet model
resnet_model = models.resnet18(pretrained=False)
num_ftrs = resnet_model.fc.in_features
resnet_model.fc = nn.Linear(num_ftrs, 5)
resnet_model.load_state_dict(torch.load('knee_osteoarthritis_resnet18.pth'))
resnet_model = resnet_model.to(device)

# Load the pre-trained DenseNet model
densenet_model = models.densenet121(pretrained=False)
num_ftrs = densenet_model.classifier.in_features
densenet_model.classifier = nn.Linear(num_ftrs, 5)
densenet_model.load_state_dict(torch.load('knee_osteoarthritis_densenet.pth'))
densenet_model = densenet_model.to(device)

# Load the pre-trained EfficientNet model
efficientnet_model = EfficientNet.from_name('efficientnet-b0')
num_ftrs = efficientnet_model._fc.in_features
efficientnet_model._fc = nn.Linear(num_ftrs, 5)
efficientnet_model.load_state_dict(torch.load('knee_osteoarthritis_efficientnet.pth'))
efficientnet_model = efficientnet_model.to(device)

# Load the pre-trained ViT model
vit_model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=5)
vit_model.load_state_dict(torch.load('knee_osteoarthritis_vit.pth'))
vit_model = vit_model.to(device)

# Define the accuracy weights for the models
weights = {
    'resnet': 0.2,  # Replace with actual accuracy weights
    'densenet': 0.3,
    'efficientnet': 0.1,
    'vit': 0.4
}

# Function to predict using the ensemble model
def predict_ensemble(models, weights, dataloader):
    models = {name: model.eval() for name, model in models.items()}
    all_preds = []

    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)

            # Get predictions from each model
            resnet_outputs = models['resnet'](inputs)
            densenet_outputs = models['densenet'](inputs)
            efficientnet_outputs = models['efficientnet'](inputs)
            vit_outputs = models['vit'](inputs)

            # Apply softmax to get probabilities
            resnet_probs = torch.softmax(resnet_outputs, dim=1)
            densenet_probs = torch.softmax(densenet_outputs, dim=1)
            efficientnet_probs = torch.softmax(efficientnet_outputs, dim=1)
            vit_probs = torch.softmax(vit_outputs, dim=1)

            # Weighted sum of probabilities
            ensemble_probs = (weights['resnet'] * resnet_probs +
                              weights['densenet'] * densenet_probs +
                              weights['efficientnet'] * efficientnet_probs +
                              weights['vit'] * vit_probs)

            _, preds = torch.max(ensemble_probs, 1)
            all_preds.extend(preds.cpu().numpy())

    return all_preds

# Helper function to show an image
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # Pause a bit so that plots are updated

# Function to visualize a few predictions
def visualize_predictions(dataloader, model_preds):
    images_so_far = 0
    num_images = 6
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloader):
            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images // 2, 2, images_so_far)
                ax.axis('off')
                ax.set_title(f'predicted: {class_names[model_preds[images_so_far - 1]]}\nactual: {class_names[labels[j]]}')
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    return

# Define the models and their weights
models = {
    'resnet': resnet_model,
    'densenet': densenet_model,
    'efficientnet': efficientnet_model,
    'vit': vit_model
}

# Predict using the ensemble model
ensemble_preds = predict_ensemble(models, weights, dataloaders['val'])

# Visualize some predictions
visualize_predictions(dataloaders['val'], ensemble_preds)
