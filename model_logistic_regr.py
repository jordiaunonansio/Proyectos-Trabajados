import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from skimage.io import imread
from skimage.transform import resize
from PIL import Image
import torch
from torchvision import transforms


IMG_SIZE = (20, 20)

def load_images_from_directory(directory, img_size):
    images = []
    labels = []
    for filename in os.listdir(directory):
        if filename.endswith(('.png')):
            label = filename[0] 
            img_path = os.path.join(directory, filename)
            img = imread(img_path, as_gray=True)
            img_resized = resize(img, img_size, anti_aliasing=True)
            images.append(img_resized.flatten())
            labels.append(label)
    return np.array(images), np.array(labels)

train_dir = '/home/xnmaster/XNAPproject-grup_08/archive/train_lletres'
X_train, y_train = load_images_from_directory(train_dir, IMG_SIZE)
test_dir = '/home/xnmaster/XNAPproject-grup_08/archive/test_lletres'
X_test, y_test = load_images_from_directory(test_dir, IMG_SIZE)

#etiqueta fotos
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)


model = LogisticRegression(max_iter=10000, solver='lbfgs', multi_class='auto')
model.fit(X_train, y_train_encoded)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test_encoded, y_pred)
print(f'Precisión del modelo: {accuracy * 100:.2f}%')
print(classification_report(y_test_encoded, y_pred, target_names=label_encoder.classes_))


def classify_letter(image_path, model, label_encoder, img_size):
    # Cargar y procesar la imagen
    img = imread(image_path, as_gray=True)
    img_resized = resize(img, img_size, anti_aliasing=True)
    img_flattened = img_resized.flatten()
    
    # Predecir la clase de la imagen
    predicted_class_encoded = model.predict([img_flattened])[0]
    predicted_class = label_encoder.inverse_transform([predicted_class_encoded])[0]
    
    return predicted_class

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
    return classify_letter(file_path, model, label_encoder, IMG_SIZE)


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

print(predict_path('/home/xnmaster/XNAPproject-grup_08/words_segmented_test'))
