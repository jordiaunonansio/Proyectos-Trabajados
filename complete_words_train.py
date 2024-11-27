import cv2
import os
from segmentacio import extreu_lletres

def guardar_imagen_en_directorio(imagen_path, identity, directorio_destino):
    # Crear la ruta completa de la nueva carpeta
    carpeta_destino = os.path.join(directorio_destino, identity)
    
    print(f"Carpeta destino: {carpeta_destino}")

    # Crear la carpeta si no existe
    if not os.path.exists(carpeta_destino):
        os.makedirs(carpeta_destino)

    lletres, num_lletres = extreu_lletres(imagen_path)
    if len(lletres) == len(identity):
        for i, (lletra, _) in enumerate(lletres):
            nombre_archivo = os.path.join(carpeta_destino, f'{identity}{i}.png')
            print(f"Guardando letra en: {nombre_archivo}")
            cv2.imwrite(nombre_archivo, lletra)
        return lletres, num_lletres
    else:
        print(f"Error: La cantidad de letras extraídas ({len(lletres)}) no coincide con la identidad ({len(identity)}).")
        return None, 0

def create_train_completewords(csv, directorio_destino):
    # Leer el archivo CSV ignorando las cabeceras
    with open(csv, 'r') as file:
        lines = file.readlines()[1:]

    # Crear el directorio destino si no existe
    if not os.path.exists(directorio_destino):
        os.makedirs(directorio_destino)

    for line in lines:
        # Separar la línea en las columnas
        parts = line.strip().split(',')

        # Extraer la identidad y la imagen de la línea
        if len(parts) != 2:
            print(f"Error en línea: {line.strip()}")
            continue

        imagen_path, identity = parts
        imagen_path = os.path.join('/home/xnmaster/TestProject/XNAPproject-grup_08/archive/train_v2/train', imagen_path)
        
        print(f"Procesando imagen: {imagen_path} con identidad: {identity}")
        
        # Guardar la imagen en el directorio destino
        guardar_imagen_en_directorio(imagen_path, identity, directorio_destino)

# Ejemplo de uso
create_train_completewords('/home/xnmaster/TestProject/XNAPproject-grup_08/archive/written_name_train_v2.csv', '/home/xnmaster/TestProject/XNAPproject-grup_08/words_segmented_train')
