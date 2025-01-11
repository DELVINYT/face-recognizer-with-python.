import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk

class DetectorRostros:
    def __init__(self):
        # Inicializar la interfaz de usuario
        self.root = tk.Tk()
        self.root.title("Detector de Rostros")
        
        # Obtener lista de cámaras disponibles
        self.camaras_disponibles = self.obtener_camaras()
        
        # Crear elementos de la interfaz
        self.setup_ui()
        
        # Inicializar variables
        self.cap = None
        self.detector_rostros = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.ejecutando = False

    def obtener_camaras(self):
        """Busca cámaras disponibles en el sistema"""
        camaras = []
        indice = 0
        while True:
            cap = cv2.VideoCapture(indice)
            if not cap.isOpened():
                break
            camaras.append(indice)
            cap.release()
            indice += 1
        return camaras

    def setup_ui(self):
        """Configura la interfaz de usuario"""
        # Selector de cámara
        frame_selector = ttk.Frame(self.root)
        frame_selector.pack(pady=10)
        
        ttk.Label(frame_selector, text="Seleccionar Cámara:").pack(side=tk.LEFT)
        self.combo_camaras = ttk.Combobox(frame_selector, 
                                         values=[f"Cámara {i}" for i in self.camaras_disponibles])
        if self.camaras_disponibles:
            self.combo_camaras.set(f"Cámara {self.camaras_disponibles[0]}")
        self.combo_camaras.pack(side=tk.LEFT, padx=5)

        # Botones
        frame_botones = ttk.Frame(self.root)
        frame_botones.pack(pady=5)
        
        self.btn_iniciar = ttk.Button(frame_botones, text="Iniciar", command=self.iniciar_detector)
        self.btn_iniciar.pack(side=tk.LEFT, padx=5)
        
        self.btn_detener = ttk.Button(frame_botones, text="Detener", command=self.detener_detector)
        self.btn_detener.pack(side=tk.LEFT, padx=5)
        self.btn_detener.config(state='disabled')

    def iniciar_detector(self):
        """Inicia la detección de rostros"""
        if not self.ejecutando:
            # Obtener el índice de la cámara seleccionada
            camara_seleccionada = int(self.combo_camaras.get().split()[-1])
            self.cap = cv2.VideoCapture(camara_seleccionada)
            
            if not self.cap.isOpened():
                tk.messagebox.showerror("Error", "No se pudo abrir la cámara seleccionada")
                return
            
            self.ejecutando = True
            self.btn_iniciar.config(state='disabled')
            self.btn_detener.config(state='normal')
            self.combo_camaras.config(state='disabled')
            self.detectar_rostros()

    def detener_detector(self):
        """Detiene la detección de rostros"""
        self.ejecutando = False
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        self.btn_iniciar.config(state='normal')
        self.btn_detener.config(state='disabled')
        self.combo_camaras.config(state='normal')

    def detectar_rostros(self):
        """Proceso principal de detección de rostros"""
        if self.ejecutando:
            ret, frame = self.cap.read()
            if ret:
                # Convertir a escala de grises
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Detectar rostros
                rostros = self.detector_rostros.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(30, 30)
                )
                
                # Dibujar rectángulos y mostrar nivel de confianza
                for (x, y, w, h) in rostros:
                    # Calcular nivel de confianza basado en el tamaño del rostro
                    # y la claridad de la detección
                    confianza = self.calcular_confianza(gray[y:y+h, x:x+w])
                    
                    # Dibujar rectángulo
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    # Mostrar porcentaje
                    texto = f"Confianza: {confianza:.1f}%"
                    cv2.putText(frame, texto, (x, y-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Mostrar frame
                cv2.imshow('Detector de Rostros', frame)
                
                # Verificar si se presiona 'q' para salir
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.detener_detector()
                else:
                    self.root.after(10, self.detectar_rostros)

    def calcular_confianza(self, roi_gray):
        """Calcula el nivel de confianza de la detección"""
        # Calcular la varianza de los píxeles como medida de claridad
        varianza = np.var(roi_gray)
        # Normalizar la varianza a un porcentaje (0-100)
        confianza = min(100, (varianza / 1000) * 100)
        return confianza

    def ejecutar(self):
        """Inicia la aplicación"""
        self.root.mainloop()

if __name__ == "__main__":
    app = DetectorRostros()
    app.ejecutar()