import cv2
import face_recognition as fr

# cargar imagenes

foto_control = fr.load_image_file("FotoA.jpg")
foto_prueba = fr.load_image_file("FotoC.jpg")

# pasar imagenes a rgb
foto_control = cv2.cvtColor(foto_control, cv2.COLOR_BGR2RGB)
foto_prueba = cv2.cvtColor(foto_prueba, cv2.COLOR_BGR2RGB)


# localizar cara control

lugar_cara_A = fr.face_locations(foto_control)[0]
cara_codificada_A = fr.face_encodings(foto_control)[0]
lugar_cara_B = fr.face_locations(foto_prueba)[0]
cara_codificada_B = fr.face_encodings(foto_prueba)[0]

# mostrar rectangulo

cv2.rectangle(foto_control,
              (lugar_cara_A[3],lugar_cara_A[0]),
               (lugar_cara_A[1],lugar_cara_A[2]),
               (0,255,0),
               2)

cv2.rectangle(foto_prueba,
              (lugar_cara_B[3],lugar_cara_B[0]),
               (lugar_cara_B[1],lugar_cara_B[2]),
               (0,255,0),
               2)

# realizar comparacion

resultado = fr.compare_faces([cara_codificada_A],cara_codificada_B)
# medida de distancia
resultado = str(resultado).replace("]"," ").replace("_"," ")
resultado = resultado.split(".")[-1].strip()
distancia = fr.face_distance([cara_codificada_A],cara_codificada_B)
distancia = float(distancia.round(2)[0])
# mostrar resultado

cv2.putText(foto_prueba,
            f"{resultado} {distancia}",
            (50,50),
            cv2.FONT_ITALIC,
            1,
            (0,255,0),
            2)

# mostrar imagenes

cv2.imshow('Foto control',foto_control)
cv2.imshow("Foto prueba", foto_prueba)


# mantener el programa abierto
cv2.waitKey(0)

