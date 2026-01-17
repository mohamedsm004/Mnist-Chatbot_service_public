import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint

# 1. Chargement et prétraitement des données
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalisation
x_train = x_train.astype(np.float32) / 255
x_test = x_test.astype(np.float32) / 255

# Redimensionnement pour le CNN (ajout du canal de couleur)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# Conversion des étiquettes en vecteurs "one-hot"
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

# 2. Construction du modèle
model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3), input_shape=(28,28,1), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dropout(0.25))
model.add(Dense(10, activation='softmax'))

# Compilation
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks pour sauvegarder le meilleur modèle
es = EarlyStopping(monitor='val_accuracy', min_delta=0.01, patience=4, verbose=1)
mc = ModelCheckpoint("best_model.h5", monitor='val_accuracy', verbose=1, save_best_only=True)
cb = [es, mc]

# Entraînement
model.fit(x_train, y_train, epochs=50, validation_split=0.3, callbacks=cb)