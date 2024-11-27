from segmentacio import extreu_lletres
import cv2
import csv
import os
import shutil

labeldict = {}
lletrescount = {
    "A": 0,
    "B": 0,
    "C": 0,
    "D": 0,
    "E": 0,
    "F": 0,
    "G": 0,
    "H": 0,
    "I": 0,
    "J": 0,
    "K": 0,
    "L": 0,
    "M": 0,
    "N": 0,
    "O": 0,
    "P": 0,
    "Q": 0,
    "R": 0,
    "S": 0,
    "T": 0,
    "U": 0,
    "V": 0,
    "W": 0,
    "X": 0,
    "Y": 0,
    "Z": 0
}
# csv to dict not reading first line
with open('archive/written_name_train_v2.csv') as f:
    next(f)  # Skip the first line
    for line in f:
        key, val = line.strip().split(',')
        labeldict[key] = val

# Vaciar el directorio de salida antes de comenzar
output_folder = 'archive/train_lletres'
for filename in os.listdir(output_folder):
    file_path = os.path.join(output_folder, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print(f'Failed to delete {file_path}. Reason: {e}')

n=0
for image_name in labeldict.keys():
    image_path = 'archive/train_v2/train/' + image_name
    lletresTrain, nlletresTrain = extreu_lletres(image_path)
    nlletresLabel = len(labeldict[image_name])

    if nlletresTrain == nlletresLabel:
        if labeldict[image_name].isalpha(): # if the label is only letters
            for i in range(nlletresTrain):
                lletraTrain = lletresTrain[i][0]
                lletraLabel = labeldict[image_name][i]

                # si hi ha menys de 500 mostres de la lletra
                if lletrescount[lletraLabel] <= 500:

                    lletra_path = 'archive/train_lletres/' + lletraLabel + str(lletrescount[lletraLabel]) +".png"
                    cv2.imwrite(lletra_path, cv2.cvtColor(lletraTrain, cv2.COLOR_BGR2RGB))
                    lletrescount[lletraLabel] += 1
        
                n+=1

                # si n es divisible entre 100
                if n % 500 == 0:
                    print(n)
                    
                # si todas las letras han sido extraidas al menos 100 veces (excluyendo los últimos 3 valores del diccionario)
                if all(value >= 499 for value in lletrescount.values()):
                    break        

