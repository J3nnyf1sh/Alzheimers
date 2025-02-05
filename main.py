import tensorflow as tf
import os
import numpy as np
import tensorflow as tf

# Use these imports from tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.callbacks import EarlyStopping


print('TensorFlow version:', tf.__version__)

trainImageDir = 'datasets/brain_tumours/train'

testImageDir = 'datasets/brain_tumours/test'

def loadImage(path):
    img = tf.keras.utils.load_img(path)  # Resize images
    img = tf.keras.utils.img_to_array(img) / 255.0  # Normalize
    return img

def loadFiles(path):
    gliomaImages = np.array([os.path.join(path, 'Glioma', img) for img in os.listdir(os.path.join(path, 'Glioma'))])
    meningiomaImages = np.array([os.path.join(path, 'Meningioma', img) for img in os.listdir(os.path.join(path, 'Meningioma'))])
    pituitaryImages = np.array([os.path.join(path, 'Pituitary', img) for img in os.listdir(os.path.join(path, 'Pituitary'))])
    healthyImages = np.array([os.path.join(path, 'NoTumour', img) for img in os.listdir(os.path.join(path, 'NoTumour'))])

    return np.array(
        [loadImage(img) for img in gliomaImages] +
        [loadImage(img) for img in meningiomaImages] +
        [loadImage(img) for img in pituitaryImages] +
        [loadImage(img) for img in healthyImages]      
    ), np.array(
            ["Glioma"] * len(gliomaImages) +
            ["Meningioma"] * np.len(meningiomaImages) +
            ["Pituitary"] * np.len(pituitaryImages) +
            ["NoTumour"] * np.len(healthyImages)
        )

trainImages, trainClassification = loadFiles(trainImageDir)
testImages, testClassification = loadFiles(testImageDir)


earlyStopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    verbose=1,
    restore_best_weights=True
)

model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(32, (5,5), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),

    tf.keras.layers.Conv2D(32, (5,5), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),

    tf.keras.layers.Conv2D(64, (4,4), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),

    tf.keras.layers.Conv2D(64, (4,4), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),

    tf.keras.layers.Conv2D(256, (2,2), activation='relu'),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.25),

    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer = 'adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.fit(
    trainImages,
    trainClassification,
    epochs = 5,
    testation_data = (testImages, testClassification),
    callbacks = earlyStopping,
    verbose = 1 
)

model.evaluate(testImages, testClassification, verbose=2)

