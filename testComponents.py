import tensorflow as tf
print(tf.__version__)  # Debería mostrar 2.16.1
print(tf.config.list_physical_devices('GPU'))  # Verifica si detecta la GPU