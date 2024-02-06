import read_sudoku_graella
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import imutils
import random
import os


#Funcio per aconseguir les cantonades de la graella
def preparar_contorns(img):    
    preparada = read_sudoku_graella.preparaimg(img)
    contorns = read_sudoku_graella.detectacontorns(preparada)
    cont_fin = read_sudoku_graella.cantonades(contorns)
    return cont_fin


#Crea un rectangle que enmarca la graella 
def crea_rectangle(cont_fin):
    # Calcula el rectangle d'area mínima que enmarca la graella
    rectangle = cv2.minAreaRect(cont_fin) 
    (x, y), (width, height), angle = rectangle
    # Calcula les coordenadas de les cantonades d'aquest rectangle
    coord_rectangle = cv2.boxPoints(rectangle)
    coord_rectangle = np.int0(coord_rectangle) 
    return coord_rectangle, width, height


def draw_rectangle(img, coord_rectangle):
    # El primer 0 es pel tipus de contorn, el 255 pel color verd i el 2 per l'amplada del contorn
    cv2.drawContours(img, [coord_rectangle], 0, (0, 255, 0), 2)  


def flat_image (img, cont_fin, width, height):
    punts_desti = [[0, 0], [width, 0], [width, height], [0, height]] 
    # Obtenir la matriu de transformació de la perspectiva
    m = cv2.getPerspectiveTransform(np.float32(cont_fin), np.float32(punts_desti)) 
    # Aplicar la transformació de la perspectiva a la copia de la imatge original
    out = img.copy()
    out = cv2.warpPerspective(out, m, (int(width), int(height)), flags=cv2.INTER_LINEAR)
    out = cv2.flip(out, 1)
    return out


def Aplanar_imatge(img):
    cont_fin = preparar_contorns(img)
    coord_rectangle, width, height = crea_rectangle(cont_fin)
    draw_rectangle(img, coord_rectangle)
    final_image = flat_image(img, cont_fin, width, height)
    return final_image
