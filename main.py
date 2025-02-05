import ssl

ssl._create_default_https_context = ssl._create_unverified_context


import tensorflow as tf
import os
import numpy as np
import tensorflow as tf

# Use these imports from tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications import MobileNetV2

from sklearn.utils import shuffle

print('TensorFlow version:', tf.__version__)

def loadImage(path):
    try:
        img = load_img(path, target_size=(128, 128))  # Resize images
        img = img_to_array(img) / 255.0  # Normalize
        return img
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

def loadFiles(path):
    gliomaImages = np.array([os.path.join(path, 'Glioma', img) for img in os.listdir(os.path.join(path, 'Glioma'))][:500] )
    meningiomaImages = np.array([os.path.join(path, 'Meningioma', img) for img in os.listdir(os.path.join(path, 'Meningioma'))][:500] )
    pituitaryImages = np.array([os.path.join(path, 'Pituitary', img) for img in os.listdir(os.path.join(path, 'Pituitary'))][:500] )
    healthyImages = np.array([os.path.join(path, 'NoTumor', img) for img in os.listdir(os.path.join(path, 'NoTumor'))][:200] )

    images = np.array(
        [loadImage(img) for img in gliomaImages] +
        [loadImage(img) for img in meningiomaImages] +
        [loadImage(img) for img in pituitaryImages] +
        [loadImage(img) for img in healthyImages]      
    )

    classification = np.array(
        [0] * len(gliomaImages) +
        [1] * len(meningiomaImages) +
        [2] * len(pituitaryImages) +
        [3] * len(healthyImages)
    )

    return images, classification


"""

"""

trainImageDir = 'brain_tumours/Train'
testImageDir = 'brain_tumours/Test'

trainImages, trainClassification = loadFiles(trainImageDir)
testImages, testClassification = loadFiles(testImageDir)

trainImages, trainClassification = shuffle(trainImages, trainClassification)
testImages, testClassification = shuffle(testImages, testClassification)

""" 
    Code used when not using a pretrained model, but less acuracy, higher loss, and slower
"""

#model = tf.keras.models.Sequential([
    ##Conv2D(32, (4,4), activation='relu'),
    ##MaxPooling2D(pool_size=(2,2)),

    ##Conv2D(64, (3,3), activation='relu'),
    ##MaxPooling2D(pool_size=(2,2)),

    ##Conv2D(128, (2,2), activation='relu'),
    ##MaxPooling2D(pool_size=(2,2)),

    ##Conv2D(256, (1,1), activation='relu'),
    ##MaxPooling2D(pool_size=(2,2)),

    #Conv2D(256, (2,2), activation='relu'),

    #Dropout(0.4),

    #Flatten(),

    #Dense(128, activation='relu'),

    #Dropout(0.3),

    #Dense(4, activation='softmax')
#])

"""
    Using a pretrained model
    - more accuracy and less loss
    - slightly faster leaning rate
"""
baseModel = MobileNetV2(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
baseModel.trainable = False  

model = tf.keras.models.Sequential([
    baseModel,

    Flatten(),

    Dense(128, activation='relu'),

    Dropout(0.3),

    Dense(4, activation='softmax')
])

print ("compiling")

model.compile(
    optimizer = 'adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

earlyStopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    verbose=1,
    restore_best_weights=True
) 

model.fit(
    trainImages,
    trainClassification,
    epochs = 50,
    validation_data = (testImages, testClassification),
    callbacks = earlyStopping,
    verbose = 1 
)


model.evaluate(testImages, testClassification, verbose=2)

test_image_path = 'brain_tumours/Test/Glioma/Glioma_test_2.jpg'  # Change to an actual test image path

# Load and preprocess the image
img = load_img(test_image_path, target_size=(128, 128))  # Resize to match the model input size
img_array = img_to_array(img) / 255.0  # Normalize
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

test_image_path2 = 'brain_tumours/Test/NoTumor/NoTumor_test_25.jpg' 

img2 = load_img(test_image_path2, target_size=(128, 128))  # Resize to match the model input size
img_array2 = img_to_array(img2) / 255.0  # Normalize
img_array2 = np.expand_dims(img_array2, axis=0) 

classification = {
    0: "Glioma",
    1: "Meningioma",
    2: "Pituitary",
    3: "NoTumour"
}

prediction = model.predict(img_array)
predicted_class_index = np.argmax(prediction)  

predicted_label = classification[predicted_class_index]  # Convert back to label

print(f"Predicted Class: {predicted_label}")

prediction2 = model.predict(img_array2)
predicted_class_index2 = np.argmax(prediction2)  

predicted_label2 = classification[predicted_class_index2]  # Convert back to label

print(f"Predicted Class: {predicted_label2}")