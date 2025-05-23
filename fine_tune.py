import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers, callbacks
import numpy as np
import os

def load_feedback_data(model_name):
    """Carga datos de retroalimentación"""
    data_path = f"feedback/{model_name}_feedback_data.npy"
    labels_path = f"feedback/{model_name}_feedback_labels.npy"
    
    if not os.path.exists(data_path) or not os.path.exists(labels_path):
        print(f"No se encontraron datos de retroalimentación para {model_name}")
        return None, None
    
    data = np.load(data_path)
    labels = np.load(labels_path)
    return data, labels

def fine_tune_model(model_name):
    """Realiza fine-tuning del modelo con datos de retroalimentación"""
    # Cargar modelo existente
    model_path = f"models/{model_name}_model.h5"
    if not os.path.exists(model_path):
        print(f"Modelo {model_path} no encontrado")
        return
    
    model = tf.keras.models.load_model(model_path)
    
    # Cargar datos de retroalimentación
    x_feedback, y_feedback = load_feedback_data(model_name)
    if x_feedback is None or y_feedback is None:
        return
    
    # Configurar callbacks
    callbacks_list = [
        callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        callbacks.ModelCheckpoint(
            f"models/{model_name}_model_finetuned.h5",
            save_best_only=True,
            monitor='loss'
        )
    ]
    
    # Compilar modelo con una tasa de aprendizaje más baja
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # Tasa baja para fine-tuning
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Entrenar con datos de retroalimentación
    model.fit(
        x_feedback, y_feedback,
        batch_size=32,
        epochs=10,
        callbacks=callbacks_list,
        verbose=1
    )
    
    # Guardar modelo actualizado
    model.save(f"models/{model_name}_model.h5")
    print(f"Modelo {model_name} actualizado con fine-tuning")

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    os.makedirs("feedback", exist_ok=True)
    
    print("Fine-tuning modelo MNIST...")
    fine_tune_model('mnist')
    
    print("\nFine-tuning modelo EMNIST...")
    fine_tune_model('emnist')