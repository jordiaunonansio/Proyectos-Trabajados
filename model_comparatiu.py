import cv2
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, accuracy_score
from save_results import plot_letter_accuracy, plot_accuracy_per_epoch, plot_loss_per_epoch, plot_confusion_matrix, save_train_test_accuracy_plot, save_train_test_loss_plot
from torchvision import transforms

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

class ImprovedCNN(nn.Module):
    def __init__(self):
        super(ImprovedCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(32)
        self.conv5 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(16 * 3 * 3, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 26)  # 26 classes for each letter

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        x = F.relu(self.bn5(self.conv5(x)))
        #print(f'After conv5 and pool: {x.shape}')
        x = self.pool(x)
        x = x.view(-1, 16 * 3 * 3)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Definir batch_size
batch_size = 18

# Crear TensorDataset i DataLoader
train_dataset = TensorDataset(train_images, train_labels)
test_dataset = TensorDataset(test_images, test_labels)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# Define the model, criterion, and optimizer
model = ImprovedCNN()
criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(model.parameters(), lr=0.001)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Variables per guardar les mètriques
train_loss_per_epoch = []
train_accuracy_per_epoch = []
test_loss_per_epoch = []
test_accuracy_per_epoch = []
all_preds = []
all_labels = []

# Training loop
num_epochs = 55
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    correct_train = 0
    total_train = 0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    avg_train_loss = train_loss / len(train_loader)
    train_accuracy = 100 * correct_train / total_train
    train_loss_per_epoch.append(avg_train_loss)
    train_accuracy_per_epoch.append(train_accuracy)
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%')

    # Evaluation loop
    model.eval()
    test_loss = 0.0
    correct_test = 0
    total_test = 0
    epoch_preds = []
    epoch_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()
            
            # Guardar prediccions i etiquetes per la matriu de confusió
            epoch_preds.extend(predicted.cpu().numpy())
            epoch_labels.extend(labels.cpu().numpy())
    
    all_preds.extend(epoch_preds)
    all_labels.extend(epoch_labels)

    avg_test_loss = test_loss / len(test_loader)
    test_accuracy = 100 * correct_test / total_test
    test_loss_per_epoch.append(avg_test_loss)
    test_accuracy_per_epoch.append(test_accuracy)
    print(f'Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')

# Guardar el model entrenat
torch.save(model.state_dict(), 'model_CNN/results/dataset3/trained_model_comparatiu.pth')

# Verificació del càlcul de la precisió
accuracy = accuracy_score(all_labels, all_preds)
print(f"Calculated Test Accuracy: {accuracy:.2f}")

# Generar matriu de confusió
conf_matrix = confusion_matrix(all_labels, all_preds)

# Plotejar la matriu de confusió amb matplotlib
plt.figure(figsize=(10, 8))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(abc))
plt.xticks(tick_marks, abc.keys(), rotation=90)
plt.yticks(tick_marks, abc.keys())

# Escriure els números a les cel·les
thresh = conf_matrix.max() / 2.
for i, j in np.ndindex(conf_matrix.shape):
    plt.text(j, i, format(conf_matrix[i, j], 'd'),
             horizontalalignment="center",
             color="white" if conf_matrix[i, j] > thresh else "black")

plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# Funcions de ploteig (suposant que ja estan definides)
plot_letter_accuracy(all_preds, all_labels, abc)
plot_accuracy_per_epoch(train_accuracy_per_epoch, "train")
plot_loss_per_epoch(train_loss_per_epoch, "train")
plot_accuracy_per_epoch(test_accuracy_per_epoch, "test")
plot_loss_per_epoch(test_loss_per_epoch, "test")
plot_confusion_matrix(all_preds, all_labels, abc)
save_train_test_accuracy_plot(train_accuracy_per_epoch, test_accuracy_per_epoch)
save_train_test_loss_plot(train_loss_per_epoch, test_loss_per_epoch)

print('Finished Training')