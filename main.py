import tensorflow as tf
import numpy as np

print('TensorFlow version:', tf.__version__)

brain_tumour = tf.keras.datasets.brain_tumour

(train_images, train_labels), (valid_images, valid_labels) = brain_tumour.load_data()
train_images, valid_images = train_images/255, valid_images/255

earlyStopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    verbose=1,
    restore_best_weights=True
)

model = tf.keras.models.Sequential([

    tf.keras.layers.Conv2D(32, (5,5), activation='relu'),
    tf.keras.layers.MaxPooling(pool_size=(2,2)),

    tf.keras.layers.Conv2D(32, (5,5), activation='relu'),
    tf.keras.layers.MaxPooling(pool_size=(2,2)),

    tf.keras.layers.Conv2D(64, (4,4), activation='relu'),
    tf.keras.layers.MaxPooling(pool_size=(2,2)),

    tf.keras.layers.Conv2D(64, (4,4), activation='relu'),
    tf.keras.layers.MaxPooling(pool_size=(2,2)),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling(pool_size=(2,2)),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling(pool_size=(2,2)),

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
    train_images,
    train_labels,
    epochs = 5,
    validation_data = (valid_images, valid_labels),
    callbacks = earlyStopping,
    verbose = 1 
)

model.evaluate(valid_images,  valid_labels, verbose=2)

