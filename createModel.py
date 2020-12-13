import numpy as np
import matplotlib.pyplot as plt

import glob
import os.path as path
from matplotlib.pyplot import imread

import os
import PIL
import PIL.Image
import tensorflow as tf

import keras
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.preprocessing.image import img_to_array, load_img, array_to_img, ImageDataGenerator
from keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model

from sklearn.model_selection import train_test_split

def plot_loss_curve(history):

    import matplotlib.pyplot as plt

    plt.figure(figsize=(15, 10))

    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')
    plt.show()   

def load_data():
    data_generator = ImageDataGenerator(rescale = 1./255)

    train_generator = data_generator.flow_from_directory(
        './images',
        target_size=(300,300),
        class_mode='categorical',
        batch_size=32,
        shuffle=True,
    )

    X, y = train_generator.next()

    print(X.shape , y.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    
    model = Sequential([
                Input(shape=(300,300,3), name='input_layer'),
                
                Conv2D(16, kernel_size=(3,3), activation='relu', name='conv_layer1'),
                MaxPooling2D(pool_size=(2,2)),
                Dropout(0.25),

                Conv2D(32, kernel_size=(3,3), activation='relu', name='conv_layer2'),
                MaxPooling2D(pool_size=(2,2)),
                Dropout(0.25),

                Conv2D(64, kernel_size=(3,3), activation='relu', name='conv_layer3'),
                MaxPooling2D(pool_size=(2,2)),
                Dropout(0.25),

                Conv2D(128, kernel_size=(3,3), activation='relu', name='conv_layer4'),
                MaxPooling2D(pool_size=(2,2)),
                Dropout(0.25),

                Conv2D(256, kernel_size=(3,3), activation='relu', name='conv_layer5'),
                MaxPooling2D(pool_size=(2,2)),
                Dropout(0.25),

                Flatten(),
                
                Dense(3, activation='softmax', name='output_layer')
            ])

    model.summary()    
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=10
    )
    plot_loss_curve(history.history)
    print(history.history)
    print("train loss=", history.history['loss'][-1])
    print("validation loss=", history.history['val_loss'][-1])    
    
    model.save('model-201511279.model')
    
    return model

def predict_image_sample(model, X_test, y_test, test_id):
    test_sample_id = test_id
        
    test_image = X_test[test_sample_id]
    
    plt.imshow(test_image)
    
    test_image = test_image.reshape(1,300,300,3)

    y_actual = y_test[test_sample_id]
    print("y_actual number=", y_actual)
    
    y_pred = model.predict(test_image)
    print("y_pred=", y_pred)
    y_pred = np.argmax(y_pred, axis=1)[0]
    print("y_pred number=", y_pred)
    plt.show()

X_train, X_test, y_train, y_test = load_data()
# train_model(X_train, y_train)
model = load_model('model-201511279.model')
for i in range(5):
    predict_image_sample(model, X_test, y_test, i)
