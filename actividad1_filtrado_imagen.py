"""
Filtrado de imagen para detección de letras centrales de una placa de automóvil usando OpenCV y Tesseract.

Autor: Paulina Méndez López - A01644629
"""
# Importaciones externas
import cv2
import numpy as np
import pytesseract
# Ruta de Tesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Cargar imagen
placa1 = cv2.imread("placa_q.jpg")
placa2 = cv2.imread("placa_2.jpg")

# Achicar imagen para facilitar lectura
placa2_reducida = placa2[::3, ::3]

def filtro_gaussiano(imagen, tamaño):
    """
    Aplica un filtro gaussiano 2D manual para suavizar la imagen.

    Parámetros:
        imagen (np.array): Imagen original.
        tamaño (int): Tamaño del kernel.

    Retorna:
        np.array: Imagen filtrada.
    """
    sig = (tamaño-1)/3
    # Generar vector gaussiano 1D
    d = np.arange(-(tamaño-1)/2, (tamaño-1)/2+1).astype(int)           
    s = np.exp(-(np.power(d,2))/(2*np.power(sig,2)))
    # Crear kernel gaussiano 2D como producto exterior
    kernel = np.zeros((tamaño, tamaño))
    for yi in range(tamaño):             
        for xi in range(tamaño):
            kernel[yi, xi]=s[yi]*s[xi]
    kernel /= kernel.sum()
    # Aplicar convolución con el kernel
    imagen_filtrada = cv2.filter2D(imagen, -1, kernel)
    return imagen_filtrada

# Esta función convierte la imagen de la placa a HSV para facilitar la deteccion de colores oscuros
def hsv_black_mask(imagen):
    """
    Convierte la imagen a HSV para facilitar la 
    deteccion de colores oscuros.

    Args:
        imagen (np.array): Imagen en formato BGR

    Returns:
        np.array: Máscara HSV para tonos negros
    """
    hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
    # Rango HSV para tonos negros
    lower = np.array([0, 0, 0])
    upper = np.array([180, 0, 114])
    # Aplicar máscara para detectar tonos negros
    mask_hsv = cv2.inRange(hsv, lower, upper)
    black_mask_hsv = cv2.bitwise_and(hsv, hsv, mask=mask_hsv)
    return black_mask_hsv

def umbral(imagen):
    """
    Convierte la imagen a escala de grises y aplica umbral inverso 
    utilizando el método de Otsu para resaltar texto oscuro sobre fondo claro.
    
    Parámetros:
        imagen (np.array): Imagen en formato BGR.

    Retorna:
        np.array: Imagen binarizada con umbral.
    """
    gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    _, binaria = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binaria

def ocr(binaria, config):
    """
    Aplica OCR a una imagen binarizada utilizando Tesseract 
    para extraer texto.

    Parámetros:
        binaria (np.array): Imagen binarizada que contiene el texto.
        config (str): Configuración personalizada para Tesseract.

    Retorna:
        str: Texto reconocido en la imagen.
    """
    texto = pytesseract.image_to_string(binaria, config=config)
    print("Texto detectado en la placa:", texto)
    return texto
    
# Aplicar filtro gaussiano para reducir el ruido
imagen_suavizada1 = filtro_gaussiano(placa1, 28)
imagen_suavizada2 = filtro_gaussiano(placa2_reducida, 15)

# Generar máscara HSV para resaltar caracteres oscuros en la primera placa
img_mask = hsv_black_mask(imagen_suavizada1)

# Aplicar umbral binario a ambas imágenes suavizadas
binaria1 = umbral(img_mask)
binaria2 = umbral(imagen_suavizada2)

# Configuración de OCR: limitar reconocimiento a caracteres válidos
config = '--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ'
config2 = '--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-'

# Aplicar OCR a las imagenes binarizadas
texto1 = ocr(binaria1, config)
texto2 = ocr(binaria2, config2)

# Mostrar resultados finales 
cv2.imshow("PlacaQ", binaria1)
cv2.imshow("Placa2", binaria2)
cv2.waitKey(0)
cv2.destroyAllWindows()