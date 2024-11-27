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
    for j in range(0, 100):
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

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

# Definició del model CRNN
class CRNN(nn.Module):
    def __init__(self, num_classes):
        super(CRNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 32x14x14
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 64x7x7
        )
        self.rnn = nn.LSTM(16*7, 64, bidirectional=True, num_layers=2, batch_first=True)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.cnn(x)
        batch_size, channels, height, width = x.size()
        x = x.permute(0, 2, 3, 1).contiguous().view(batch_size, width, -1)  # Reorganitza per a la capa recurrent
        x, _ = self.rnn(x)
        x = self.fc(x[:, -1, :])  # Utilitza la sortida de l'últim pas de temps
        return x

# Inicialització del model, funció de pèrdua i optimitzador
num_classes = len(abc)
model = CRNN(num_classes=num_classes)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Dispositiu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Entrenament del model
num_epochs = 17
train_loss_per_epoch = []
train_accuracy_per_epoch = []
test_loss_per_epoch = []
test_accuracy_per_epoch = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()
    
    train_loss_per_epoch.append(running_loss/len(train_loader))
    train_accuracy_per_epoch.append(100 * correct_train / total_train)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}, Accuracy: {100 * correct_train / total_train}%")

    model.eval()
    running_loss_test = 0.0
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss_test += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()
    
    test_loss_per_epoch.append(running_loss_test/len(test_loader))
    test_accuracy_per_epoch.append(100 * correct_test / total_test)
    print(f"Test Loss: {running_loss_test/len(test_loader)}, Test Accuracy: {100 * correct_test / total_test}%")

# Guardar el model
torch.save(model.state_dict(), 'model_CRNN/results/dataset2/trained_model_CRNN.pth')    #La comento perquè vull provar poque epochs

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

plot_letter_accuracy(all_preds, all_labels, abc)
plot_accuracy_per_epoch(train_accuracy_per_epoch, "train")
plot_loss_per_epoch(train_loss_per_epoch, "train")
plot_accuracy_per_epoch(test_accuracy_per_epoch, "test")
plot_loss_per_epoch(test_loss_per_epoch, "test")
plot_confusion_matrix(all_preds, all_labels, abc)
save_train_test_accuracy_plot(train_accuracy_per_epoch, test_accuracy_per_epoch)
save_train_test_loss_plot(train_loss_per_epoch, test_loss_per_epoch)
