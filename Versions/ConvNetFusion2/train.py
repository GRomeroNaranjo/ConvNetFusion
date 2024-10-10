import torch
from torch import nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms
from dataclasses import dataclass
import torch.functional as F


transform = transforms.Compose([
    transforms.ToTensor()
])

cifar10 = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
bird_images = [img for img, label in cifar10 if label == 2]

image_size = bird_images[0].numel()
bird_images_tensor = torch.stack([img.flatten() for img in bird_images])
device = torch.device("cpu")
y_train = bird_images_tensor.to(device)
X_train = torch.randn((len(bird_images), image_size)).to(device)


class Config(dataclass):
    image_size = 32 * 32
    sound_size = 32 * 32
    input_colour_dim = 1

    X_train = X_train
    y_train = y_train
    epochs = 50
    learning_rate = 3e-4

class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        #Convolutional Layers
        self.conv1 = nn.Conv2d(config.input_colour_dim, 16, 3, 1)
        self.r_1 = nn.ReLU()

        self.conv1 = nn.Conv2d(16, 32, 3, 1)
        self.r_2 = nn.ReLU()

        self.conv1 = nn.Conv2d(32, 64, 3, 1)
        self.r_3 = nn.ReLU()

        self.conv1 = nn.Conv2d(64, 128, 3, 1)
        self.r_4 = nn.ReLU()

        #Dense Layer
        self.l_1 = nn.Linear(16 * 16 * 128, 13824)
        self.r_5 = nn.ReLU()

        self.l_2 = nn.Linear(13824, 9216)
        self.r_6 = nn.ReLU()

        self.l_3 = nn.Linear(9216, 4608)
        self.r_7 = nn.ReLU()

        self.l_4 = nn.Linear(4608, config.image_size)
        self.r_8 = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.r_1(x)

        x = self.conv1(x)
        x = self.r_2(x)

        x = self.conv1(x)
        x = self.r_3(x)

        x = self.conv1(x)
        x = self.r_4(x)

        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)

        x = self.l_1(x)
        x = self.r_5(x) 

        x = self.l_2(x) 
        x = self.r_6(x)

        x = self.l_3(x) 
        x = self.r_7(x) 

        x = self.l_4(x)
        x = self.r_8(x)

        return x
    

class Model:
    def __init__(self, config):
        self.model = Model(config)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.config = config
        self.epochs = config.epochs
        self.learning_rate = config.learning_rate

    def train(self, x):
        for epoch in range(self.epochs):
            y_pred = self.model(x)
            loss = self.criterion(y_pred, self.config.y_train)

            loss.backward()
            self.optimizer.step()

            print(f"""
            Model Report:
                Configurations:
                    - Learning Rate: {self.learning_rate}
                    - Optimizer: Adam
                    - Mini-Batch: False
                    - Criterion: MSELoss
                    - Simultaneous Distillation: False
                Functionality Report:
                    - Epochs: {epoch} / {self.epochs}
                    - Loss: {loss}
            """)
                