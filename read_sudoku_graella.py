import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import imutils
import random
import os

#Deteccion matriz grande sudoku
def preparaimg(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  #Convierte la imagen a escala de grises
    image = cv2.GaussianBlur(image, (5,5), 0)  #Aplica un filtro Gaussiano para reducir el ruido
    (thresh, image) = cv2.threshold(image, 0, 255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)  # Binarizacion de la imagen
    return image
'''
def identify_edges(image):
    edges = cv2.Canny(image, 75, 100)
    return edges'''

def detectacontorns(image):
    return cv2.findContours(image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # Encuentra los contornos de los objetos en la imagen

def cantonades(contours):
    contours = sorted(contours[0], key=cv2.contourArea, reverse=True)[:10]  # Ordena los contornos por su área de mayor a menor y toma los 10 más grandes
    for c in contours:
        perimetre = cv2.arcLength(c, True)  # Calcula el perímetro del contorno
        aproximacio = cv2.approxPolyDP(c, 0.02*perimetre, True)  # Convierte la forma del contorno que ha detectado en una forma geometrica mas "comun" en este caso un cuadrado
        if len(aproximacio) == 4:  # Si el polígono tiene 4 vértices, se considera que es el contorno del tablero de Sudoku
            return aproximacio

def retallacontorns(image, contours):
    x, y, ancho, altura = cv2.boundingRect(contours) # Encuentra los límites del contorno del Sudoku
    #x,y guardan el vertice superior izquierdo como referencia de donde empieza el contorno
    cropped_image = image[y:y+altura, x:x+ancho] # Recorta la imagen utilizando los límites del contorno
    return cropped_image

def draw_contours(image, contours):
    cv2.drawContours(image, [contours], -1, (0,255,0), 3) 

