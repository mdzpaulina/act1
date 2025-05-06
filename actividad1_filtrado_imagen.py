#Filtrado de Imagen para Detección de Letras Centrales de una Placa de Automóvil en Python
#Paulina Méndez López - A01644629
import cv2
import numpy as np
import pytesseract
# Ruta de Tesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Cargar imagen
placa1 = cv2.imread("placa_q.jpg")
placa2 = cv2.imread("placa_2.jpg")

#Achicar la imagen para poder leerla más facilmente
placa2_reducida = placa2[::3, ::3]

# Filtro Gaussiano
def filtro_gaussiano(imagen, tamaño):
    # Se define un tamaño de kernel
    sig = (tamaño-1)/3
    # Se genera una curva gausiana para obtener los valores del filtro
    d = np.arange(-(tamaño-1)/2, (tamaño-1)/2+1).astype(int)
    s = np.exp(-(np.power(d,2))/(2*np.power(sig,2)))
    # Se obtienen los coeficientes del filtro mediante el producto 2D de la curva
    kernel = np.zeros((tamaño, tamaño))
    for yi in range(tamaño):
        for xi in range(tamaño):
            kernel[yi, xi]=s[yi]*s[xi]
    kernel /= kernel.sum()
    imagen_filtrada = cv2.filter2D(imagen, -1, kernel)
    
    return imagen_filtrada

def hsv_black_mask(imagen):
    # Se convierte la imagen de la placa a HSV para facilitar la deteccion de colores oscuros
    hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
    #Definir los rangos para la mascara
    lower = np.array([0, 0, 0])
    upper = np.array([180, 0, 114])
    #Se crea una mascara para detectar tonos negros
    mask_hsv = cv2.inRange(hsv, lower, upper)
    black_mask_hsv = cv2.bitwise_and(hsv, hsv, mask=mask_hsv)
    return black_mask_hsv

def umbral(imagen):
    # Convertir a escala de grises para facilitar la aplicación del umbral
    gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    # Umbral para invertir texto oscuro sobre fondo claro y que tesseract pueda leer bien las letras
    _, binaria = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binaria

#Se aplica OCR a las imagenes con umbral para obtener el texto reconocido
def ocr(binaria, config):
    texto = pytesseract.image_to_string(binaria, config=config)
    print("Texto detectado en la placa:", texto)
    return texto
    

#Se suaviza la imagen para reducir ruido y no se tomen otros caracteres innecesarios en el OCR
imagen_suavizada1 = filtro_gaussiano(placa1, 28)
imagen_suavizada2 = filtro_gaussiano(placa2_reducida, 15)

img_mask = hsv_black_mask(imagen_suavizada1)

binaria1 = umbral(img_mask)
binaria2 = umbral(imagen_suavizada2)

# OCR: restringimos a mayúsculas y caracteres si solo hay letras
config = '--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ'
config2 = '--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-'

texto1 = ocr(binaria1, config)
texto2 = ocr(binaria2, config2)

#Mostramos el resultado de la imagen final 
cv2.imshow("PlacaQ", binaria1)
cv2.imshow("Placa2", binaria2)
cv2.waitKey(0)
cv2.destroyAllWindows()