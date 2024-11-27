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
from model import SimpleCNN

abc = {
    "A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "H": 7, "I": 8, "J": 9, "K": 10, "L": 11, "M": 12, "N": 13, "O": 14, "P": 15, "Q": 16, "R": 17, "S": 18, "T": 19, "U": 20, "V": 21, "W": 22, "X": 23, "Y": 24, "Z": 25
}
model = SimpleCNN()
model.load_state_dict(torch.load('model_CNN/results/dataset2/trained_model.pth'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
model.eval()

def predict_letter(lletra):
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


def predict_path(path):
    count_nota = 0
    count_total = 0
    for root, dirs, folders in os.walk(path):
        for nom in dirs:
            for r, img, f in os.walk(root+'/'+nom):
                predictions = []
                if len(f) != 0:
                    f = sorted(f)
                    for foto in f:
                        file_path = os.path.join(root, nom+'/'+foto)
                        predicted_letter = predict_letter_from_file(file_path)
                        predictions.append(predicted_letter)
                    count_total +=1
                    if ''.join(predictions) == nom:
                        count_nota +=1
    return count_nota/count_total*100

print(predict_path('/home/xnmaster/TestProject/XNAPproject-grup_08/words_segmented_test'))
