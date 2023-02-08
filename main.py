# por el terminal externo funciona perfecto
# import cv2
# img = cv2.imread("./image.jpg")
# print(img)
## pip install dlib-19.22.99-cp39-cp39-win_amd64.whl
''' https://github.com/tobyyosoba777/Computer-Vision-OpenCv-Python/blob/main/Libraries/dlib-19.22.99-cp39-cp39-win_amd64.whl
'''
import os
import cv2
import face_recognition
import numpy as np
from deepface import DeepFace

# Variables de caras encodadas y nombres de las caras
encodings_caras = []
nombres_caras = []

# Directorio actual
directorio = os.getcwd()
path_caras = os.path.join(directorio, "caras/")

# Obtiene lista de fotos de caras en el directorio
fotos_caras = [f for f in os.listdir(path_caras) if f.endswith(".jpg")]

# Entrena caras
for foto in fotos_caras:
    # Carga imagen de cara y obtiene encoding
    imagen = face_recognition.load_image_file(os.path.join(path_caras, foto))
    encoding = face_recognition.face_encodings(imagen)[0]
    encodings_caras.append(encoding)
    nombres_caras.append(foto)

# Inicia captura de video
video_capture = cv2.VideoCapture(0)

while True:
    # Obtiene frame de video
    ret, frame = video_capture.read()
    # Redimensiona el frame
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    # Convierte a RGB
    rgb_small_frame = small_frame[:, :, ::-1]

    # Encuentra locaciones de caras y encodings en el frame
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    nombres_detectados = []

    # Compara encodings de caras detectadas con encodings entrenadas
    for encoding in face_encodings:
        matches = face_recognition.compare_faces(encodings_caras, encoding)
        nombre = "Prohibido->Desconocido"
        distancias_cara = face_recognition.face_distance(encodings_caras, encoding)
        mejor_coincidencia = np.argmin(distancias_cara)

        if matches[mejor_coincidencia]:
            nombre = nombres_caras[mejor_coincidencia]
        nombres_detectados.append(nombre)
        ## aqui la emocion y for para != lineas

    # Dibuja rectángulos y nombres en el frame
    for (top, derecha, abajo, izquierda), nombre in zip(
        face_locations, nombres_detectados
    ):
        top *= 4
        derecha *= 4
        abajo *= 4
        izquierda *= 4

        cv2.rectangle(frame, (izquierda, top), (derecha, abajo), (0, 0, 255), 2)

        cv2.rectangle(
            frame, (izquierda, abajo - 35), (derecha, abajo), (0, 0, 255), cv2.FILLED
        )

    fuente = cv2.FONT_HERSHEY_DUPLEX
    try:
        cv2.putText(
        frame, "Acceso" + nombre, (izquierda + 6, abajo - 6), fuente, 1.0, (255, 255, 255), 1
    )
    
        
    # Muestra frame con caras reconocidas
        cv2.imshow("Video", frame)
    # Si se presiona 'q', sale del ciclo
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    except:
        print("Pongase enfrente de la camara")
    finally:
        pass