import cv2
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from save_results import plot_letter_accuracy, plot_accuracy_per_epoch, plot_loss_per_epoch, plot_confusion_matrix, save_train_test_accuracy_plot, save_train_test_loss_plot
import cv2
from PIL import Image
from torchvision import transforms
import torch

# Preparació dels Generadors d'Imatges
train_dir = 'archive/train_lletres'
validation_dir = 'archive/test_lletres'

# Definim la mida fixa a la que redimensionarem les imatges
fixed_size = (28, 28)

# Llegir imatges train
abc = {
    "A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "H": 7, "I": 8, "J": 9, "K": 10, "L": 11, "M": 12, "N": 13, "O": 14, "P": 15, "Q": 16, "R": 17, "S": 18, "T": 19, "U": 20, "V": 21, "W": 22, "X": 23, "Y": 24, "Z": 25
}
imatges_train = []
train_labels = []

imatges_test = []
test_labels = []

for lletra in abc.keys():
    for j in range(0, 500):
        imatge = cv2.imread(f'{train_dir}/{lletra}{j}.png', cv2.IMREAD_GRAYSCALE)
        imatge = cv2.resize(imatge, fixed_size)
        imatges_train.append(imatge)
        train_labels.append(abc[lletra])

for lletra in abc.keys():
    for j in range(0, 160):
        imatge = cv2.imread(f'{validation_dir}/{lletra}{j}.png', cv2.IMREAD_GRAYSCALE)
        imatge = cv2.resize(imatge, fixed_size)
        imatges_test.append(imatge)
        test_labels.append(abc[lletra])

# Convertir les llistes de Python a tensors de PyTorch
train_images = torch.tensor(np.array(imatges_train), dtype=torch.float32).unsqueeze(1) / 255.0
train_labels = torch.tensor(train_labels, dtype=torch.long)
test_images = torch.tensor(np.array(imatges_test), dtype=torch.float32).unsqueeze(1) / 255.0
test_labels = torch.tensor(test_labels, dtype=torch.long)

# Crear TensorDataset i DataLoader
train_dataset = TensorDataset(train_images, train_labels)
test_dataset = TensorDataset(test_images, test_labels)

train_loader = DataLoader(train_dataset, batch_size=18, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=18, shuffle=True)

# Creació del Model CNN simple amb softmax
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
        self.fc = nn.Linear(8 * 3 * 3, 26)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 8 * 3 * 3)
        x = self.fc(x)
        return x

model = SimpleCNN()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Definir la funció de pèrdua i l'optimitzador
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Definició de les funcions d'entrenament i test
def train(model, loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch_features, batch_labels in loader:
        batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)

        optimizer.zero_grad()
        outputs = model(batch_features)
        train_loss = criterion(outputs, batch_labels)
        train_loss.backward()
        optimizer.step()
        running_loss += train_loss.item()

        _, predicted = torch.max(outputs, 1)
        total += batch_labels.size(0)
        correct += (predicted == batch_labels).sum().item()

    loss = running_loss / len(loader)
    accuracy = 100 * correct / total
    print("Train loss = {:.6f}, Train accuracy = {:.2f}%".format(loss, accuracy))
    return accuracy, loss

def test(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_features, batch_labels in loader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
            outputs = model(batch_features)
            test_loss = criterion(outputs, batch_labels)
            running_loss += test_loss.item()

            _, predicted = torch.max(outputs, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()

    loss = running_loss / len(loader)
    accuracy = 100 * correct / total
    print("Test loss = {:.6f}, Test accuracy = {:.2f}%".format(loss, accuracy))
    return accuracy, loss


# Entrenar el model

train_accuracy_per_epoch= []
train_loss_per_epoch = []
test_accuracy_per_epoch = []
test_loss_per_epoch = []

epochs = 55
for epoch in range(epochs):
    print(f'Epoch {epoch + 1}/{epochs}')
    train_accuracy, train_loss = train(model, train_loader, optimizer, criterion)
    accuracy, loss = test(model, test_loader, criterion)
    train_accuracy_per_epoch.append(train_accuracy)
    train_loss_per_epoch.append(train_loss)
    test_accuracy_per_epoch.append(accuracy)
    test_loss_per_epoch.append(loss)

# Guardar el model entrenat
torch.save(model.state_dict(), 'model_CNN/results/dataset2/trained_model.pth')    #La comento perquè vull provar poque epochs

# Guardar els resultats
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

def predict_letter(lletra):
    model.eval()
    with torch.no_grad():
        lletra = lletra.to(device)
        outputs = model(lletra)
        _, predicted = torch.max(outputs.data, 1)
        predicted = predicted.cpu().numpy()
        # invertimos el diccionario abc para mapear de índices a letras
        inv_abc = {v: k for k, v in abc.items()}
        return inv_abc[predicted[0]]

def predict_letter_from_file(file_path):
    # Cargamos la imagen y la convertimos a escala de grises
    img = Image.open(file_path).convert('L')

    # Definimos las transformaciones que queremos aplicar a la imagen
    transform = transforms.Compose([
        transforms.Resize((28, 28)),  # Asumimos que tu modelo espera imágenes de 28x28
        transforms.ToTensor(),
    ])

    # Aplicamos las transformaciones a la imagen y añadimos una dimensión extra
    # para representar el batch size
    lletra = transform(img).unsqueeze(0)

    # Llamamos a predict_letter con el tensor
    return predict_letter(lletra)

# Usamos la función con el path al archivo de la imagen
'''predicted_letter = predict_letter_from_file('/home/xnmaster/TestProject/XNAPproject-grup_08/archive/test_lletres/Z296.png')
print(f"La letra predicha es: {predicted_letter}")'''

plot_letter_accuracy(all_preds, all_labels, abc)
plot_accuracy_per_epoch(train_accuracy_per_epoch, "train")
plot_loss_per_epoch(train_loss_per_epoch, "train")
plot_accuracy_per_epoch(test_accuracy_per_epoch, "test")
plot_loss_per_epoch(test_loss_per_epoch, "test")
plot_confusion_matrix(all_preds, all_labels, abc)
save_train_test_accuracy_plot(train_accuracy_per_epoch, test_accuracy_per_epoch)
save_train_test_loss_plot(train_loss_per_epoch, test_loss_per_epoch)



print('Finished Training')
