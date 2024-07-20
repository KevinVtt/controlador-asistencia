import cv2
import face_recognition as fr
import os
import numpy
from datetime import datetime

def codificar(imagenes):
    # crear una lista nueva
    lista_codificar = []
    for imagen in imagenes:
        # codificamos la imagen
        lista_codificar.append(fr.face_encodings(imagen)[0])

    # retornamos
    return lista_codificar

# registrar los ingresos

def registrar_ingresos(persona):
    f = open('registros.csv','r+')
    lista_datos = f.readline()
    nombre_registro = []
    for linea in lista_datos:
        ingreso = linea.split(",")
        nombre_registro.append(ingreso[0])
    
    if persona not in nombre_registro:
        ahora = datetime.now()
        string_ahora = ahora.strftime("%H:%M:%S")
        f.writelines(f"\n{persona}, {string_ahora}")


def mostrar_imagenes(imagenes,empleados):
    contador = 0
    for imagenes in mis_imagenes:
        cv2.imshow(empleados[contador],imagenes)
        contador+=1

def verificar_yo_mismo(empleados_codificados, nombre_empleados):
    # Verificar yo mismo
    foto_mia = fr.load_image_file("D:\CursoPython\Dia 14\Asistencia\FotoMia.jpg")
    foto_mia = cv2.cvtColor(foto_mia, cv2.COLOR_BGR2RGB)
    foto_mia_codificada = fr.face_encodings(foto_mia)[0]

    coincidencias_encontradas = False

    for empleado_codificado, nombre_empleado in zip(empleados_codificados, nombre_empleados):
        coincidencias = fr.compare_faces([empleado_codificado], foto_mia_codificada)
        distancias = fr.face_distance([empleado_codificado], foto_mia_codificada)
        print(f"Distancia con {nombre_empleado}: {distancias[0]}")

        if distancias[0] <= 0.6:
            print(f"¡Coincide con {nombre_empleado}!")
            coincidencias_encontradas = True
            break

    if not coincidencias_encontradas:
        print("No coincide con ningún empleado")




# crear base de datos 
ruta = 'Fotos_Empleados'
mis_imagenes = []
nombre_empleados = []
lista_empleados = os.listdir(ruta)



for empleado in lista_empleados:

    try:

        # ruta completa
        ruta_completa = os.path.join(ruta,empleado)
        # cargamos la foto
        foto_prueba = fr.load_image_file(ruta_completa)
        # convertirlo a rgb
        foto_prueba_rgb = cv2.cvtColor(foto_prueba,cv2.COLOR_BGR2RGB)
        # agregar a la lista
        mis_imagenes.append(foto_prueba_rgb)

        nombre_empleados.append(os.path.splitext(empleado)[0])

    except Exception as e:
        print(f"Error al procesar {empleado}: {e}")


lista_empleado_codificada = codificar(mis_imagenes)
mostrar_imagenes(mis_imagenes,nombre_empleados)

# verificar_yo_mismo(lista_empleado_codificada,nombre_empleados)

# tomar una imagen de camara web

captura = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# leer imagen de la camara
exito,imagen = captura.read()

if not exito:
    print("No se ha podido encontrar la captura")
else:
    # reconocer cara en captura
    cara_captura = fr.face_locations(imagen)

    # codificar la imagen

    cara_captura_codificada = fr.face_encodings(imagen,cara_captura)

    for caracodif, caraubic in zip(cara_captura_codificada,cara_captura):
        coincidencias = fr.compare_faces(lista_empleado_codificada,caracodif)
        distancias = fr.face_distance(lista_empleado_codificada,caracodif)
        print(distancias)

        indice_coincidencia = numpy.argmin(distancias)

        # mostrar coincidencia
        if distancias[indice_coincidencia] > 0.6:
            print("No coincide con ninguno de nuestros empleados")
        else:
            # encontramos al empleado
            nombre = nombre_empleados[indice_coincidencia]
            y1,x2,y2,x1 = caraubic()
            cv2.rectangle(imagen,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(imagen,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(imagen,nombre,(x1 + 6, y2 - 6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            
            registrar_ingresos(nombre)

            # mostrar la imagen obtenida 

            cv2.imshow("Imagen web",imagen)            
            cv2.waitKey(0)