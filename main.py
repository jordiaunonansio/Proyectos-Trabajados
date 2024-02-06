import read_sudoku_celles
from flat_sudoku import Aplanar_imatge as Aplanar_imatge
from solucionar_sudoku import Sudoku as Sudoku
import sys
import random
import os    
import cv2
import numpy as np
from predict_numbers import predict as predict
import matplotlib.pyplot as plt
import tensorflow as tf

#LLegim imatge
folder="Fotos_Sudoku_Propies/Propies_test"
#folder="sudoku_dataset-master/images"
foto=random.choice(os.listdir(folder))
path = (folder+'/'+foto)
path = 'Fotos_Sudoku_Propies/20230512_162724_OK.jpg'
#if the path passed exists and is an image
try:
    sudoku_a = cv2.imread(sys.argv[1])
except:
    sudoku_a = cv2.imread(path)


sudoku_a = cv2.imread(path)

#Detectem graella
quad = read_sudoku_celles.Detecta_graella(sudoku_a) #deteccio graella sense aplanar

plt.figure()
plt.imshow(quad)
plt.show()

quadricula_plana = Aplanar_imatge(quad) #aplanar la deteccio feta anteriorment
retalls = read_sudoku_celles.retalla_celles(quadricula_plana) #retall de les celles a la graella aplanada

plt.figure()
plt.imshow(quadricula_plana)
plt.show()

#Creacio i omplir la matriu de 9x9
matriu=np.zeros((9,9))
plt.imshow(retalls[0])# exemple de cel·la
plt.show()

for i in range(9):
    for j in range(9):
        #plt.imshow(retalls[i*9+j])
        matriu[i,j]=list(predict(retalls[i*9+j]))[0]
        print(matriu[i,j])

#Resolucio sudoku
print(matriu, 'abans de resoldre')
Solucio = Sudoku(matriu)
print(Solucio, 'despres de resoldre')
