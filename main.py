# por el terminal externo funciona perfecto
#import cv2
#img = cv2.imread("./image.jpg")
#print(img)

import face_recognition
import cv2
import numpy as np
import os
import glob

encodings_caras = []
nombre_de_caras = []
directorio = os.getcwd()
path = os.path.join(directorio, 'caras/')
Lista_de_fotos = [f for f in glob.glob(path+'*.jpg')]
Numero_de_Fotos = len(Lista_de_fotos)
nombres = Lista_de_fotos.copy()

# Entrena caras
for i in range(Numero_de_Fotos):
    globals()['image_{}'.format(i)] = face_recognition.load_image_file(Lista_de_fotos[i])
    globals()['image_encoding_{}'.format(i)] = face_recognition.face_encodings(globals()['image_{}'.format(i)])[0]
    encodings_caras.append(globals()['image_encoding_{}'.format(i)])

# Crea Matriz de nombres conocidos
    nombres[i] = nombres[i].replace(directorio, "")  
    nombre_de_caras.append(nombres[i])
    
    print(nombres[i])
    print(nombre_de_caras)

face_locations = []
face_encodings = []
face_nombres = []
process_this_frame = True

video_capture = cv2.VideoCapture(0)
while True:
    ret, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]

    if process_this_frame:
        face_locations = face_recognition.face_locations( rgb_small_frame)
        face_encodings = face_recognition.face_encodings( rgb_small_frame, face_locations)
    
    face_nombres = []

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces (encodings_caras, face_encoding)
        name = "Unknown"

    face_distances = face_recognition.face_distance( encodings_caras, face_encoding)

    best_match_index = np.argmin(face_distances)

    if matches[best_match_index]:
        name = nombre_de_caras[best_match_index]
        face_nombres.append(name)
    
    process_this_frame = not process_this_frame

# Display the results

    for (top, right, bottom, left), name in zip(face_locations, face_nombres):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
# Draw a rectangle around the face
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

# Input text label with a name below the face
    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

# Display the resulting image
    cv2.imshow('Video', frame)
# Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        pass
