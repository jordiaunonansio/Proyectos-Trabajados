import cv2
import numpy as np
import matplotlib.pyplot as plt
import read_sudoku_graella
import math
import tensorflow as tf
import imutils
import random
import os
import flat_sudoku

def Detecta_graella(img): # Recive la imagen leida tal cual del directorio sin ninguna modificacion
    img_preparada = read_sudoku_graella.preparaimg(img)
    contorns = read_sudoku_graella.detectacontorns(img_preparada)
    cont_fin = read_sudoku_graella.cantonades(contorns)
    graella = read_sudoku_graella.retallacontorns(img, cont_fin)
    return graella

def retalla_celles(graella):
    alto, ancho, _ = graella.shape
    ancho_recorte = math.ceil(ancho / 9)
    alto_recorte = math.ceil(alto / 9)

    # Lista para almacenar los recortes
    recortes = []

    # Recorrer la imagen y hacer los recortes
    for y in range(0, alto, alto_recorte):
        for x in range(0, ancho, ancho_recorte):
            recorte = graella[y:y+alto_recorte, x:x+ancho_recorte]
            recortes.append(recorte)
    return recortes












'''def detectar_lineas_sudoku(img):
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    suavizado = cv2.GaussianBlur(gris, (7, 7), 0)
    bordes = cv2.Canny(suavizado, 10, 70, 3)
    lineas = cv2.HoughLinesP(bordes, 1, np.pi/180, threshold=150, minLineLength=1600, maxLineGap=1700)
    set = []
    for linea in lineas:
        x1, y1, x2, y2 = linea[0]
        set.append([x1, y1, x2, y2])
    return set

def find_intersection(line1, line2):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    if denominator == 0:
        # Las líneas son paralelas o coincidentes
        return None

    # Calcular las coordenadas del punto de intersección
    intersection_x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denominator
    intersection_y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denominator

    return intersection_x, intersection_y

def detectar_lineas_sudoku(img):
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    suavizado = cv2.GaussianBlur(gris, (7, 7), 0)
    bordes = cv2.Canny(suavizado, 10, 70, 3)
    lineas = cv2.HoughLinesP(bordes, 1, np.pi/180, threshold=150, minLineLength=1600, maxLineGap=1700)
    conjunto_puntos = []
    for linea in lineas:
        x1, y1, x2, y2 = linea[0]
        conjunto_puntos.append([x1, y1, x2, y2])

    puntos_interseccion = []
    for i in range(len(conjunto_puntos)):
        for j in range(i + 1, len(conjunto_puntos)):
            punto_interseccion = find_intersection(conjunto_puntos[i], conjunto_puntos[j])
            if punto_interseccion is not None:
                puntos_interseccion.append(punto_interseccion)

    puntos_interseccion = sorted(puntos_interseccion, key=lambda x: (x[1], x[0]))

    cuadros_recortados = []
    tamano_cuadro = 100
    for punto in puntos_interseccion:
        x, y = punto
        x_inicio = x - tamano_cuadro // 2
        y_inicio = y - tamano_cuadro // 2
        x_fin = x_inicio + tamano_cuadro
        y_fin = y_inicio + tamano_cuadro
        cuadro_recortado = img[int(y_inicio):int(y_fin), int(x_inicio):int(x_fin)].copy()
        cuadros_recortados.append(cuadro_recortado)

    return cuadros_recortados
'''