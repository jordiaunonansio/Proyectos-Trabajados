import tensorflow as tf
import os
import cv2
import imghdr
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from itertools import chain
import shutil

def clean_dataframe(df):
    df_cleaned = df[df['IDENTITY'] != 'UNREADABLE']  # Eliminar filas con 'UNREADABLE' en la columna especificada
    df_cleaned = df_cleaned.dropna()  # Eliminar filas con valores nulos
    return df_cleaned

import cv2
import numpy as np
import matplotlib.pyplot as plt

def extreu_lletres(imatge_path, min_size=3, desired_size=15):
    
    # Llegir la imatge
    imatge = cv2.imread(imatge_path)
    
    # Convertir a escala de grisos
    gray = cv2.cvtColor(imatge, cv2.COLOR_BGR2GRAY)
   
    
    # Binaritzar la imatge
    _, binaritzada = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    
    # Trobar contorns
    contorns, _ = cv2.findContours(binaritzada, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    lletres_imatges = []
    
    for contorn in contorns:
        x, y, w, h = cv2.boundingRect(contorn)
        
        # Filtrar contorns que son menors que el tamaño mínimo establecido
        if w < min_size or h < min_size:
            continue
        
        lletra_imatge = imatge[y:y+h, x:x+w]
        max_side = max(w, h)
        
        # Crear una imatge quadrada amb fons blanc
        square_imatge = np.ones((max_side, max_side, 3), dtype=np.uint8) * 255
        
        if w > h:
            y_offset = (max_side - h) // 2
            square_imatge[y_offset:y_offset+h, :w] = lletra_imatge
        else:
            x_offset = (max_side - w) // 2
            square_imatge[:h, x_offset:x_offset+w] = lletra_imatge
        
        # Escalar la imagen a un tamaño deseado para asegurar visibilidad
        square_imatge = cv2.resize(square_imatge, (desired_size, desired_size))
        
        lletres_imatges.append((square_imatge, x))

    # Ordenar las imágenes de letras por su posición x
    lletres_imatges.sort(key=lambda img: img[1])
    
    # Ordenar las imágenes de letras por su posición horizontal
    #lletres_imatges.sort(key=lambda img: cv2.boundingRect(cv2.findContours(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0])[0])
    
    return lletres_imatges, len(lletres_imatges)

def mostra_lletres(imatge_path, output_folder):
    # Asegurarse de que el directorio de salida existe, y si no, crearlo
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    else:
        # Vaciar el directorio de salida antes de comenzar
        for filename in os.listdir(output_folder):
            file_path = os.path.join(output_folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')

    lletres_imatges = extreu_lletres(imatge_path)[0]
    
    for i, lletra_imatge in enumerate(lletres_imatges):
        # Guardar cada imagen en el directorio especificado
        filename = f"{output_folder}/lletra_{i}.png"
        cv2.imwrite(filename, cv2.cvtColor(lletra_imatge[0], cv2.COLOR_BGR2RGB))
    
    print(f"Imágenes guardadas en {output_folder}")

# Exemple d'ús de la funció
mostra_lletres('archive/test_v2/test/TEST_0009.jpg', 'res_detect')
mostra_lletres('archive/test_v2/test/TEST_0091.jpg', 'res_detect')


