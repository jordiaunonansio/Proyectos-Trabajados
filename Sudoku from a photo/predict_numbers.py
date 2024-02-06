import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import imutils
import random
import os

model = tf.keras.models.load_model('digits.model')

def probar_test():
    #carreguem les dades test
    numbers_test = []
    labels_test = []
    path = 'numbers/test'
    for img in os.listdir(path):
        if img.endswith('.png'):
            img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
            numbers_test.append(img_array)
            labels_test.append(int(img[-5]))
    labels_test = np.array(labels_test)

    correcte = 0
    path = 'numbers/test'
    for i in range(0,100):
        img = os.listdir(path)[i]
        img = numbers_test[i].reshape(1, 28, 28, 1)
        if int(labels_test[i])== np.argmax(model.predict(img)):
            correcte+=1

    print('Accuracy: '+ str(correcte/100))



def predict(img):
    #img = tf.keras.utils.normalize(img, axis=1)
    # show
    img = convert_to_28_28(img)
    # plt.imshow(img, cmap='gray')
    # plt.show()
    pred = model.predict(img.reshape(1, 28, 28, 1))
    pred, prob = np.argmax(pred), np.max(pred)
    return pred, prob   
 
def convert_to_28_28(image):
    
    #image = cv2.bitwise_not(image)
    image = cv2.resize(image, (38, 38))
    #gray
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image.reshape(38, 38)
    #cut 5 px of borders
    image = image[5:33, 5:33]
    return image

def predict_celles():
    #llegir les imatges de la carpeta celles
    path = 'celles'
    correcte = 0
    for img in os.listdir(path):
        if img.endswith('.jpg'):
            img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)  
            # plt.imshow(img_array)
            # plt.show()
            
            pred, prob = predict(img_array)
            print('Predicted: '+str(pred)+', Probability: '+str(prob))

