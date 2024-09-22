import torch
from torchvision import datasets, transforms
from torch import nn
import torch.optim as optim
import matplotlib.pyplot as plt

class ToVertical:
    def __call__(self, img):
        return img.view(-1, 1)
    
transform = transforms.Compose([
    transforms.ToTensor(),
    ToVertical()
])

cifar10 = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
cat_images = [img for img, label in cifar10 if label == 3]

image_size = cat_images[0].numel()
bird_images_tensor = torch.stack([img.flatten() for img in cat_images])
device = torch.device("cpu")
y_train = bird_images_tensor.to(device)
X_train = torch.randn((len(cat_images), image_size)).to(device)

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
    
num = 2
for _ in range(7):
    num = num * 2
    sample_input = X_train[1].unsqueeze(0)
    model1 = RunModel(num, 0.001)
    model1.train(sample_input, y_train)
    output1 = model1.model(X_train[1])
    output1 = output1.view(32, 32, 3)
    output1_np = output1.squeeze().detach().numpy()

    plt.imshow(output1_np)
    plt.colorbar()
    plt.show()
    
