import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers, callbacks
import numpy as np
import os

def load_dataset(dataset='mnist'):
    """Carga MNIST, EMNIST Balanced, o EMNIST Letters"""
    if dataset == 'mnist':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        num_classes = 10
        needs_transpose = False
    elif dataset == 'emnist_balanced':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.emnist.load_data('balanced')
        num_classes = 47
        needs_transpose = True
    elif dataset == 'emnist_letters':
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.emnist.load_data('letters')
        y_train = y_train - 1 # Etiquetas 1-26 a 0-25
        y_test = y_test - 1  # Etiquetas 1-26 a 0-25
        num_classes = 26 # A-Z
        needs_transpose = True
    else:
        raise ValueError(f"Dataset desconocido: {dataset}")

    if needs_transpose:
        x_train = np.transpose(x_train, (0, 2, 1))
        x_test = np.transpose(x_test, (0, 2, 1))
    
    x_train = x_train.astype('float32') / 255.0
    x_train = x_train.reshape((-1, 28, 28, 1))
    x_test = x_test.astype('float32') / 255.0
    x_test = x_test.reshape((-1, 28, 28, 1))
    
    x_train = 1.0 - x_train # Invertir colores: tinta negra, fondo blanco
    x_test = 1.0 - x_test   # Invertir colores: tinta negra, fondo blanco
    
    return (x_train, y_train), (x_test, y_test), num_classes

def create_model(input_shape, num_classes):
    """Crea un modelo CNN mejorado"""
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
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

def train_and_save(model_name_key, model_save_filename):
    """Entrena y guarda el modelo."""
    print(f"--- Entrenando modelo para: {model_name_key} ---")
    (x_train, y_train), (x_test, y_test), num_classes = load_dataset(model_name_key)
    print(f"Datos cargados: {x_train.shape[0]} muestras de entrenamiento, {num_classes} clases.")

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        fill_mode='nearest'
    )
    
    model = create_model((28, 28, 1), num_classes)
    
    model_path_to_save = f"models/{model_save_filename}.h5"

    callbacks_list = [
        callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=5, min_lr=1e-5),
        callbacks.ModelCheckpoint(
            model_path_to_save,
            save_best_only=True,
            monitor='val_accuracy'
        )
    ]
    
    print(f"Iniciando entrenamiento... Modelo se guardará en {model_path_to_save}")
    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=128),
        validation_data=(x_test, y_test),
        epochs=50, # Puedes ajustar
        callbacks=callbacks_list,
        verbose=1
    )
    
    print(f"Cargando el mejor modelo guardado desde: {model_path_to_save} para evaluación final.")
    # Asegurarse de que el modelo se haya guardado antes de intentar cargarlo
    if os.path.exists(model_path_to_save):
        best_model = tf.keras.models.load_model(model_path_to_save)
        test_loss, test_acc = best_model.evaluate(x_test, y_test, verbose=0)
        print(f"\nPrecisión en test ({model_name_key}): {test_acc:.4f}")
    else:
        print(f"Error: El modelo no se guardó correctamente en {model_path_to_save}. Evaluando el modelo actual en memoria (puede no ser el mejor).")
        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
        print(f"\nPrecisión en test ({model_name_key}) (modelo en memoria): {test_acc:.4f}")
        
    print(f"--- Entrenamiento para {model_name_key} completado. Modelo guardado como {model_path_to_save} ---\n")

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    
    # --- PARA ENTRENAR SOLO EL MODELO DE LETRAS (A-Z) ---
    # Descomenta las otras líneas si quieres volver a entrenar los otros modelos después.
    
    # print("Entrenamiento de MNIST (números) OMITIDO.")
    # # train_and_save(model_name_key='mnist', model_save_filename='mnist_model')
    
    # print("Entrenamiento de EMNIST Balanced (números y letras) OMITIDO.")
    # # train_and_save(model_name_key='emnist_balanced', model_save_filename='emnist_model') 
    
    print(">>> Iniciando entrenamiento EXCLUSIVO para EMNIST Letters (A-Z) <<<")
    train_and_save(model_name_key='emnist_letters', model_save_filename='emnist_letters_model')
    print(">>> Entrenamiento de EMNIST Letters (A-Z) finalizado. <<<")