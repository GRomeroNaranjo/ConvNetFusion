import torch
import matplotlib.pyplot as plt
from model.train import Model, X_train

model = Model()
model.load_state_dict(torch.load('model_parameters.pth'))

sample_input = X_train[1].unsqueeze(0)
output = model(sample_input)

output = output.view(32, 32, 3)
output_np = output.squeeze().detach().numpy()

plt.imshow(output_np)
plt.colorbar()
plt.show()
