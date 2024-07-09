import torch 
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torch.nn import nn

from torchvision.models import resnet50

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
dataset = ImageFolder(root="dataset 2", transform=transform)

# dataloader for batches   
dataloader = transforms.DataLoader(dataset, batch_size= 32, shuffle=True)

# load pre-train model
model = resnet50(pretrained=True)

# change the input layer dimension 
input_dim = len(dataset.classes)
model.fc = nn.Linear(model.fn.in_features, input_dim )

# freeze some layers
for para in model.parameters():
    para.requires_grad= False

# unfreeze last layers for fine-tuning 


