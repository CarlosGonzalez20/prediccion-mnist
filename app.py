import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import numpy as np
import cv2
from PIL import Image, ImageGrab # Asegúrate de tener ImageGrab
import tensorflow as tf
import os
import io # Movido al inicio correctamente

class DigitRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Reconocedor de Dígitos/Letras IA")
        self.root.geometry("400x520") # Un poco más de alto por si acaso
        
        # Cargar y compilar modelos
        self.models = {
            "MNIST": self.load_model("models/mnist_model_finetuned.h5"),
            "EMNIST Balanced": self.load_model("models/emnist_model.h5")
        }
        self.current_model = "MNIST"
        
        self.feedback_data = []
        self.feedback_labels = []
        
        self.last_processed_img = None
        self.last_predicted_class = None
        
        self.setup_ui()
        
    def load_model(self, model_path):
        if not os.path.exists(model_path):
            messagebox.showerror("Error", f"Modelo {model_path} no encontrado. Ejecute train.py primero")
            return None
            
        try:
            model = tf.keras.models.load_model(model_path)
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            print(f"Modelo {model_path} cargado y compilado correctamente")
            return model
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo cargar {model_path}:\n{str(e)}")
            return None
    
    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(expand=True, fill=tk.BOTH)
        
        model_frame = ttk.LabelFrame(main_frame, text="Modelo IA", padding="10")
        model_frame.pack(fill=tk.X, pady=5)
        
        self.model_var = tk.StringVar(value=self.current_model)
        for model_name in self.models:
            rb = ttk.Radiobutton(
                model_frame, 
                text=model_name,
                variable=self.model_var,
                value=model_name,
                command=self.change_model
            )
            rb.pack(anchor=tk.W)
        
        canvas_frame = ttk.LabelFrame(main_frame, text="Dibuje aquí", padding="10")
        canvas_frame.pack(pady=10)
        
        self.canvas = tk.Canvas(
            canvas_frame, 
            width=280, 
            height=280, 
            bg='white', 
            highlightthickness=1, 
            highlightbackground="gray"
        )
        self.canvas.pack()
        
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=5)
        
        self.btn_predict = ttk.Button(
            btn_frame, 
            text="Predecir", 
            command=self.predict
        )
        self.btn_predict.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        
        self.btn_clear = ttk.Button(
            btn_frame, 
            text="Limpiar", 
            command=self.clear_canvas
        )
        self.btn_clear.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        
        feedback_btn_frame = ttk.Frame(main_frame) # Marco separado para botones de feedback
        feedback_btn_frame.pack(fill=tk.X, pady=5)

        self.btn_correct = ttk.Button(
            feedback_btn_frame,
            text="Correcto 👍",
            command=self.feedback_correct
        )
        self.btn_correct.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        
        self.btn_incorrect = ttk.Button(
            feedback_btn_frame,
            text="Incorrecto 👎",
            command=self.feedback_incorrect
        )
        self.btn_incorrect.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=2)
        
        result_frame = ttk.LabelFrame(main_frame, text="Resultado", padding="10")
        result_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.result_label = ttk.Label(
            result_frame, 
            text="Dibuje algo y haga clic en Predecir",
            font=('Helvetica', 14),
            wraplength=350 # Para que el texto se ajuste si es largo
        )
        self.result_label.pack(pady=10)
        
        self.confidence_bar = ttk.Progressbar(
            result_frame,
            orient='horizontal',
            length=200,
            mode='determinate'
        )
        self.confidence_bar.pack(pady=5)
        
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.reset)
        self.last_x = self.last_y = None
    
    def change_model(self):
        self.current_model = self.model_var.get()
        msg = f"Modelo cambiado a {self.current_model}"
        print(msg)
        self.result_label.config(text=msg)
        self.clear_canvas()
    
    def draw(self, event):
        if self.last_x and self.last_y:
            self.canvas.create_line(
                self.last_x, self.last_y, event.x, event.y,
                width=12, fill='black', capstyle=tk.ROUND, smooth=True
            )
        self.last_x, self.last_y = event.x, event.y
    
    def reset(self, event):
        self.last_x = self.last_y = None
    
    def clear_canvas(self):
        self.canvas.delete("all")
        # self.canvas.configure(bg='white') # No es necesario reconfigurar bg cada vez
        self.result_label.config(text="Dibuje algo y haga clic en Predecir")
        self.confidence_bar['value'] = 0
        self.last_processed_img = None
        self.last_predicted_class = None

    def predict(self):
        """Realiza la predicción usando ImageGrab para capturar el canvas"""
        if not self.models[self.current_model]:
            messagebox.showerror("Error", "Modelo no disponible")
            return
            
        try:
            # 1. Capturar contenido del canvas usando ImageGrab
            self.canvas.update_idletasks() # Asegurar que el canvas esté actualizado

            x1 = self.canvas.winfo_rootx()
            y1 = self.canvas.winfo_rooty()
            
            # Usar el width/height configurado para el canvas
            # winfo_width/height pueden variar ligeramente o ser afectados por DPI
            # pero para ImageGrab necesitamos las dimensiones renderizadas.
            width = self.canvas.winfo_width()
            height = self.canvas.winfo_height()
            
            x2 = x1 + width
            y2 = y1 + height

            # print(f"Canvas screen coordinates: x1={x1}, y1={y1}, x2={x2}, y2={y2} (w={width}, h={height})")

            img_grabbed = ImageGrab.grab(bbox=(x1, y1, x2, y2))
            
            # El canvas se configuró como 280x280. Si ImageGrab da algo diferente
            # (por ej. por escalado de pantalla en algunos OS), reescalamos.
            # Usamos las dimensiones configuradas del canvas como referencia.
            canvas_configured_width = self.canvas.cget("width")
            canvas_configured_height = self.canvas.cget("height")

            if img_grabbed.size != (canvas_configured_width, canvas_configured_height):
                print(f"Advertencia: Tamaño de imagen capturada ({img_grabbed.size}) difiere del canvas configurado ({canvas_configured_width}x{canvas_configured_height}). Reescalando.")
                img_grabbed = img_grabbed.resize((int(canvas_configured_width), int(canvas_configured_height)), Image.Resampling.LANCZOS)

            img_pil = img_grabbed.convert('L') # Convertir a escala de grises
            img_np = np.array(img_pil)
            
            # Guardar imagen cruda para depuración
            cv2.imwrite(f"debug_{self.current_model}_grabbed_raw.png", img_np)
            
            # Verificar si la imagen capturada tiene contenido (tinta negra sobre fondo blanco)
            # Contamos píxeles que no son predominantemente blancos (ej. < 200)
            non_background_pixels = np.sum(img_np < 200) 
            # Ajusta este umbral (50) si es necesario. Para una imagen de 280x280,
            # 50 píxeles de tinta es muy poco. Quizás 200-500 sería más razonable.
            min_ink_pixels = (int(canvas_configured_width) * int(canvas_configured_height)) * 0.005 # 0.5% de píxeles de tinta
            if non_background_pixels < min_ink_pixels : 
                raise ValueError(f"Imagen capturada parece vacía o con muy poco dibujo (píxeles de tinta: {non_background_pixels}).")
            
            # 2. Preprocesamiento (ahora espera tinta negra sobre fondo blanco)
            self.last_processed_img = self.preprocess_image(img_np)
            
            if self.last_processed_img is None or np.any(np.isnan(self.last_processed_img)):
                raise ValueError("Imagen procesada inválida o contiene NaN")
            
            # 3. Predicción
            predictions = self.models[self.current_model].predict(self.last_processed_img, verbose=0)
            self.last_predicted_class = np.argmax(predictions[0])
            confidence = np.max(predictions[0])
            
            # 4. Mostrar resultados
            # (Tu lógica de mapeo de clases para EMNIST es correcta)
            if self.current_model == "MNIST":
                result_char = str(self.last_predicted_class)
            else: # EMNIST Balanced
                if self.last_predicted_class < 10: # 0-9
                    result_char = str(self.last_predicted_class)
                elif self.last_predicted_class < 36: # A-Z (10-35)
                    result_char = chr(self.last_predicted_class - 10 + ord('A'))
                else: # a-z (36-61 para 'balanced' que tiene 47 clases - verificar esto)
                      # EMNIST Balanced tiene 47 clases: 0-9, A-Z (mayúsculas), a-z (minúsculas - algunas fusionadas)
                      # El mapeo de EMNIST balanced es: 0-9, A-Z (10-35), a,b,d,e,f,g,h,n,q,r,t (36-46)
                      # Tu mapeo original:
                      # elif self.last_predicted_class < 36: result = chr(self.last_predicted_class + 55) -> Esto es para A-Z si 10 es A
                      # else: result = chr(self.last_predicted_class + 61) -> Esto es para a-z si 36 es a
                      # Keras EMNIST Balanced: 0-9, A-Z (10-35), a,b,d,e,f,g,h,n,q,r,t (36-46)
                      # Vamos a usar un mapeo explícito para EMNIST Balanced (47 clases)
                    
                    # Mapeo para EMNIST 'balanced' (47 clases)
                    # 0-9: Dígitos
                    # 10-35: Letras mayúsculas A-Z
                    # 36-46: Letras minúsculas (un subconjunto: a, b, d, e, f, g, h, n, q, r, t)
                    # Este es el mapeo que `tensorflow.keras.datasets.emnist.load_data('balanced')` usa internamente.
                    # La forma en que `train.py` carga EMNIST balanced no especifica un mapeo de caracteres,
                    # solo usa las etiquetas numéricas 0-46.
                    # Para mostrar, necesitamos el mapeo correcto.
                    # Las etiquetas para 'balanced' son: 0-9, A-Z (10-35), y luego algunas minúsculas.
                    # Revisando el mapeo de EMNIST 'balanced':
                    # 0-9  -> 0-9
                    # 10-35 -> A-Z
                    # 36 -> a (ord 97)
                    # 37 -> b (ord 98)
                    # 38 -> d (ord 100)
                    # 39 -> e (ord 101)
                    # 40 -> f (ord 102)
                    # 41 -> g (ord 103)
                    # 42 -> h (ord 104)
                    # 43 -> n (ord 110)
                    # 44 -> q (ord 113)
                    # 45 -> r (ord 114)
                    # 46 -> t (ord 116)

                    # Vamos a simplificar el display para EMNIST por ahora, ya que el mapeo exacto
                    # de las 11 minúsculas específicas requiere una tabla o una lógica más compleja.
                    # Tu mapeo original podría no ser exacto para EMNIST 'balanced'.
                    # Por ahora, mostremos el índice si no es un dígito o mayúscula simple.
                    
                    if self.last_predicted_class < 10: # 0-9
                        result_char = str(self.last_predicted_class)
                    elif self.last_predicted_class < 36: # A-Z (índices 10-35)
                        result_char = chr(self.last_predicted_class - 10 + ord('A'))
                    else: # Índices 36-46 para minúsculas específicas de EMNIST Balanced
                        # Para simplificar la visualización, mostramos "LC" + índice o el índice directamente
                        # Si quieres el char exacto, necesitas la tabla de mapeo para las clases 36-46
                        emnist_balanced_lowercase_map = {
                            36: 'a', 37: 'b', 38: 'd', 39: 'e', 40: 'f',
                            41: 'g', 42: 'h', 43: 'n', 44: 'q', 45: 'r', 46: 't'
                        }
                        result_char = emnist_balanced_lowercase_map.get(self.last_predicted_class, f"idx:{self.last_predicted_class}")


            self.result_label.config(text=f"Predicción: {result_char} (Confianza: {confidence:.2%})")
            self.confidence_bar['value'] = confidence * 100
            
            # Guardar imagen procesada para depuración (la 28x28 antes de añadir batch dim)
            # self.last_processed_img es (1, 28, 28, 1)
            img_to_save_for_debug = self.last_processed_img[0, :, :, 0] * 255
            cv2.imwrite(f"debug_{self.current_model}_pred_{result_char}_{self.last_predicted_class}.png", img_to_save_for_debug)
            
        except Exception as e:
            messagebox.showerror("Error en Predicción", f"Ocurrió un error:\n{str(e)}")
            self.last_processed_img = None
            self.last_predicted_class = None
            import traceback
            traceback.print_exc() # Para más detalles en la consola

    def preprocess_image(self, img_np):
        """
        Preprocesa la imagen para el modelo.
        Espera: img_np (NumPy array) de la captura del canvas (tinta negra sobre fondo blanco).
        Produce: Imagen normalizada (tinta negra 0.0, fondo blanco 1.0), redimensionada a 28x28.
        """
        try:
            # img_np ya es: tinta negra (~0), fondo blanco (~255)

            # 1. Umbralización para binarizar la imagen limpiamente
            # Tinta (valores bajos) se vuelve 0 (negro).
            # Fondo (valores altos) se vuelve 255 (blanco).
            _, processed_img = cv2.threshold(img_np, 128, 255, cv2.THRESH_BINARY)
            # Usar THRESH_BINARY_INV si la lógica de colores fuera al revés (blanco sobre negro)
            # pero aquí queremos negro sobre blanco, así que THRESH_BINARY es correcto.

            # 2. Redimensionar directamente a 28x28
            # El canvas es 280x280, ImageGrab lo captura. Lo reducimos a 28x28.
            # Esto mantiene la proporción ya que ambos son cuadrados.
            resized_img = cv2.resize(processed_img, (28, 28), interpolation=cv2.INTER_AREA)
            
            # En este punto, resized_img es tinta 0 (negro), fondo 255 (blanco).
            # El script de entrenamiento hace:
            #   original_mnist (tinta blanca 255, fondo negro 0)
            #   normalized (tinta blanca 1.0, fondo negro 0.0)
            #   inverted (tinta negra 0.0, fondo blanco 1.0) -> ESTO ESPERA EL MODELO
            
            # 3. Normalizar
            # Tinta negra (0) / 255.0 -> 0.0
            # Fondo blanco (255) / 255.0 -> 1.0
            # Esto coincide con lo que el modelo espera.
            normalized_img = resized_img.astype(np.float32) / 255.0
            
            # 4. Expandir dimensiones para el formato del modelo (batch_size, height, width, channels)
            final_img = np.expand_dims(normalized_img, axis=(0, -1))
            
            # (Opcional) Guardar imagen justo antes de retornar para depuración
            # cv2.imwrite(f"debug_{self.current_model}_preprocessed_final.png", resized_img) # Guardar la 28x28

            return final_img
        except Exception as e:
            print(f"Error en preprocesamiento: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def feedback_correct(self):
        if self.last_processed_img is None or self.last_predicted_class is None:
            messagebox.showwarning("Advertencia", "Primero realice una predicción válida.")
            return
            
        if np.any(np.isnan(self.last_processed_img)):
            messagebox.showerror("Error", "La imagen procesada contiene datos inválidos (NaN).")
            return
            
        self.feedback_data.append(self.last_processed_img[0]) # Guardar la imagen (28,28,1)
        self.feedback_labels.append(self.last_predicted_class)
        messagebox.showinfo("Éxito", f"Retroalimentación guardada como correcta: clase {self.last_predicted_class}")
        self.save_feedback()

    def feedback_incorrect(self):
        if self.last_processed_img is None or self.last_predicted_class is None:
            messagebox.showwarning("Advertencia", "Primero realice una predicción válida.")
            return
            
        if np.any(np.isnan(self.last_processed_img)):
            messagebox.showerror("Error", "La imagen procesada contiene datos inválidos (NaN).")
            return

        # Mapeo para entrada de usuario
        # EMNIST 'balanced' tiene 47 clases.
        # 0-9, A-Z (mayúsculas), y un subconjunto de 11 minúsculas: a,b,d,e,f,g,h,n,q,r,t
        prompt_detail = " (0-9"
        if self.current_model == "EMNIST Balanced":
            prompt_detail += ", A-Z, o minúsculas: a,b,d,e,f,g,h,n,q,r,t"
        prompt_detail += "):"
        
        label_str = simpledialog.askstring("Etiqueta Incorrecta", f"Ingrese la etiqueta correcta{prompt_detail}")
        
        if label_str is None: return # Usuario canceló

        corrected_label_idx = -1

        if self.current_model == "MNIST":
            if label_str.isdigit() and 0 <= int(label_str) <= 9:
                corrected_label_idx = int(label_str)
            else:
                messagebox.showerror("Error", "Etiqueta inválida para MNIST. Use 0-9.")
                return
        else: # EMNIST Balanced
            emnist_char_to_idx = {}
            # Dígitos 0-9 -> índices 0-9
            for i in range(10): emnist_char_to_idx[str(i)] = i
            # Mayúsculas A-Z -> índices 10-35
            for i in range(26): emnist_char_to_idx[chr(ord('A') + i)] = 10 + i
            # Minúsculas específicas
            lowercase_map_chars = {'a': 36, 'b': 37, 'd': 38, 'e': 39, 'f': 40, 'g': 41, 'h': 42, 'n': 43, 'q': 44, 'r': 45, 't': 46}
            emnist_char_to_idx.update(lowercase_map_chars)

            if label_str in emnist_char_to_idx:
                corrected_label_idx = emnist_char_to_idx[label_str]
            else:
                messagebox.showerror("Error", "Etiqueta inválida para EMNIST Balanced. Verifique los caracteres permitidos.")
                return
                
        self.feedback_data.append(self.last_processed_img[0]) # Guardar la imagen (28,28,1)
        self.feedback_labels.append(corrected_label_idx)
        messagebox.showinfo("Éxito", f"Retroalimentación guardada con etiqueta '{label_str}' (índice {corrected_label_idx})")
        self.save_feedback()

    def save_feedback(self):
        if not self.feedback_data or not self.feedback_labels:
            print("No hay datos de retroalimentación para guardar.")
            return
            
        try:
            # Convertir listas a arrays de NumPy antes de guardar
            # Asegurarse de que todos los elementos en feedback_data tengan la misma forma
            feedback_data_np = np.array(self.feedback_data, dtype=np.float32)
            feedback_labels_np = np.array(self.feedback_labels, dtype=np.int32) # Etiquetas son enteros
            
            if np.any(np.isnan(feedback_data_np)) or np.any(np.isnan(feedback_labels_np)):
                print("Error: Los datos de retroalimentación contienen valores NaN.")
                # Limpiar datos inválidos si es necesario o simplemente no guardar
                return 
                
            os.makedirs("feedback", exist_ok=True)
            
            # Cargar datos existentes si los hay, y añadir los nuevos
            data_file = f"feedback/{self.current_model}_feedback_data.npy"
            labels_file = f"feedback/{self.current_model}_feedback_labels.npy"

            if os.path.exists(data_file) and os.path.exists(labels_file):
                existing_data = np.load(data_file)
                existing_labels = np.load(labels_file)
                feedback_data_np = np.concatenate((existing_data, feedback_data_np), axis=0)
                feedback_labels_np = np.concatenate((existing_labels, feedback_labels_np), axis=0)

            np.save(data_file, feedback_data_np)
            np.save(labels_file, feedback_labels_np)
            
            print(f"Guardado feedback para {self.current_model}: Total {len(feedback_data_np)} muestras.")
            # Limpiar feedback en memoria después de guardar para no duplicar si se vuelve a llamar
            self.feedback_data = []
            self.feedback_labels = []

        except Exception as e:
            print(f"Error al guardar feedback: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    # Verificar si los modelos existen, solo como advertencia inicial
    if not os.path.exists("models/mnist_model.h5"):
        print("Advertencia: models/mnist_model.h5 no encontrado. Ejecute train.py primero.")
    if not os.path.exists("models/emnist_model.h5"):
        print("Advertencia: models/emnist_model.h5 no encontrado. Ejecute train.py primero.")
    
    root = tk.Tk()
    app = DigitRecognizerApp(root)
    root.mainloop()