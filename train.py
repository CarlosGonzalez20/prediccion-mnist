import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers, callbacks
import numpy as np
import os

# Resto del código permanece igual
def load_dataset(dataset='mnist'):
    """Carga MNIST o EMNIST Balanced"""
    if dataset == 'mnist':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        num_classes = 10
    else:
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.emnist.load_data('balanced')
        x_train = np.transpose(x_train, (0, 2, 1))  # Rotar imágenes EMNIST
        x_test = np.transpose(x_test, (0, 2, 1))
        num_classes = 47
    
    # Normalizar y redimensionar
    x_train = x_train.astype('float32') / 255.0
    x_train = x_train.reshape((-1, 28, 28, 1))
    x_test = x_test.astype('float32') / 255.0
    x_test = x_test.reshape((-1, 28, 28, 1))
    
    # Invertir colores (fondo negro, escritura blanca, como en el canvas)
    x_train = 1.0 - x_train
    x_test = 1.0 - x_test
    
    return (x_train, y_train), (x_test, y_test), num_classes

def create_model(input_shape, num_classes):
    """Crea un modelo CNN mejorado"""
    model = models.Sequential([
        layers.Input(shape=input_shape),
        
        # Bloque 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Bloque 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Bloque 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        # Clasificación
        layers.Flatten(),
        layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_and_save(model_name='mnist'):
    """Entrena y guarda el modelo"""
    # 1. Cargar datos
    (x_train, y_train), (x_test, y_test), num_classes = load_dataset(model_name)
    
    # 2. Data augmentation
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        fill_mode='nearest'
    )
    
    # 3. Crear modelo
    model = create_model((28, 28, 1), num_classes)
    
    # 4. Callbacks
    callbacks_list = [
        callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-5),
        callbacks.ModelCheckpoint(
            f"models/{model_name}_model.h5",
            save_best_only=True,
            monitor='val_accuracy'
        )
    ]
    
    # 5. Entrenar
    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=128),
        validation_data=(x_test, y_test),
        epochs=50,
        callbacks=callbacks_list,
        verbose=1
    )
    
    # 6. Evaluar
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nPrecisión en test ({model_name}): {test_acc:.4f}")

if __name__ == "__main__":
    # Crear directorio si no existe
    os.makedirs("models", exist_ok=True)
    
    # Entrenar modelo MNIST
    print("Entrenando modelo MNIST...")
    train_and_save('mnist')
    
    # Entrenar modelo EMNIST
    print("\nEntrenando modelo EMNIST Balanced...")
    train_and_save('emnist')