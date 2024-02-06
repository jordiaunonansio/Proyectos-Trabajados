
import tensorflow as tf
import os
import cv2
import numpy as np

#DETECCIÓ DE DIGITS
#carreguem dades de MNIST
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)



#carreguem les nostres dades
numbers_train = []
labels_train = []
path = 'numbers/train'
for img in os.listdir(path):
    if img.endswith('.png'):
        img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
        numbers_train.append(img_array)
        labels_train.append(int(img[-5]))
labels_train = np.array(labels_train)


#normalitzem les dades
numbers_train = tf.keras.utils.normalize(numbers_train, axis=1)

#només cal executar una vegada: 

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=0)


model.fit(numbers_train, labels_train, epochs=25)

model.save('digits.model')