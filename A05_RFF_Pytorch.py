#==========================================
#  Red Feed Forward en Pytorch
#==========================================
#  Julián T. Sagredo
#  Fundamentos de Inteligencia Artificial
#  Matemática Algorítmica
#  ESFM IPN 2026
#==========================================
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

#=================================
# 1. Configuración y Dispositivo
#=================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
layer_sizes = [784, 512, 256, 10]
num_epochs = 10 
batch_size = 64

#===========
# 2. Modelo
#===========
class SimpleNet(nn.Module):
    def __init__(self, layers_list):
        super(SimpleNet, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(layers_list[i], layers_list[i+1]) 
                                     for i in range(len(layers_list)-1)])

    def forward(self, x):
        x = x.view(x.size(0), -1)
        for i, layer in enumerate(self.linears):
            x = layer(x)
            if i < len(self.linears) - 1:
                x = torch.relu(x)
        return x

model = SimpleNet(layer_sizes).to(device) # <--- A LA GPU
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

#==========
# 3. Datos
#==========
transform = transforms.Compose([transforms.ToTensor()])
train_loader = DataLoader(datasets.MNIST('./data', train=True, download=True, transform=transform), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(datasets.MNIST('./data', train=False, transform=transform), batch_size=10, shuffle=True)

#==================
# 4. Entrenamiento
#==================
print(f"Entrenando en: {device}")
model.train()
for epoch in range(num_epochs):
    for data, target in train_loader:
        data, target = data.to(device), target.to(device) # <--- A LA GPU
        optimizer.zero_grad()
        loss = criterion(model(data), target)
        loss.backward()
        optimizer.step()
    print(f"Época {epoch + 1}/{num_epochs} terminada")

#===============================
# 5. Visualización (CORREGIDA)
#===============================
def visualizar_predicciones(model, loader):
    model.eval()
    data, target = next(iter(loader))
    
    # Inferencia
    with torch.no_grad():
        output = model(data.to(device))
        predicciones = output.argmax(dim=1).cpu() # Volvemos a CPU para graficar

    plt.figure(figsize=(12, 5))
    for i in range(len(data)):
        plt.subplot(2, 5, i + 1) # <--- Crea la cuadrícula
        plt.imshow(data[i].squeeze(), cmap='gray')
        
        pred, real = predicciones[i].item(), target[i].item()
        color = 'green' if pred == real else 'red'
        
        plt.title(f"IA: {pred}\nReal: {real}", color=color)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

visualizar_predicciones(model, test_loader)

