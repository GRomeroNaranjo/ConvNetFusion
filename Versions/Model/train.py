import torch
from torchvision import datasets, transforms
from torch import nn
import torch.optim as optim

class ToVertical:
    def __call__(self, img):
        return img.view(-1, 1)
    
transform = transforms.Compose([
    transforms.ToTensor(),
    ToVertical()
])

cifar10 = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
bird_images = [img for img, label in cifar10 if label == 2]

image_size = bird_images[0].numel()
bird_images_tensor = torch.stack([img.flatten() for img in bird_images])
device = torch.device("cpu")
y_train = bird_images_tensor.to(device)
X_train = torch.randn((len(bird_images), image_size)).to(device)

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.l_1 = nn.Linear(image_size, 512)
        self.r_1 = nn.ReLU()
        self.l_2 = nn.Linear(512, 768)
        self.r_2 = nn.ReLU()
        self.l_3 = nn.Linear(768, image_size)

    def forward(self, x):
        x = self.l_1(x)
        x = self.r_1(x)
        x = self.l_2(x)
        x = self.r_2(x)
        y = self.l_3(x)
        return y
    
class RunModel:
    def __init__(self, epochs, learning_rate):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.model = Model().to(device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def train(self, X_train, y_train):
        for epoch in range(self.epochs):
            self.model.train()
            output = self.model(X_train)
            loss = self.criterion(output, y_train)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
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
    
    def save_parameters(self, path):
        torch.save(self.model.state_dict(), path)
        return "Model saved successfully"
    
    def predict(self, x):
        y = self.model(x)
        return y
    
model = RunModel(100, 0.001)
model.train(X_train, y_train)
model.save_parameters("model_parameters.pth")
