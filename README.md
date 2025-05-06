# Detección de Letras en Placas Automovilísticas

Este proyecto aplica técnicas de procesamiento de imágenes utilizando OpenCV y Tesseract OCR para detectar y extraer letras centrales de placas automovilísticas. Está escrito en Python y estructurado de manera modular para facilitar su mantenimiento y expansión.

## Descripción

A partir de imágenes de placas vehiculares, el script:
- Aplica un filtro gaussiano manual para reducir ruido.
- Extrae regiones oscuras con una máscara HSV.
- Aplica umbral inverso con el método de Otsu.
- Utiliza Tesseract OCR para obtener el texto reconocido.

## Tecnologías utilizadas

- [Python 3](https://www.python.org/)
- [OpenCV](https://opencv.org/)
- [NumPy](https://numpy.org/)
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)

## Estructura del proyecto

- actividad1_filtrado_imagen.py
- README.md
- placa_q.jpg
- placa_2.jpg
- Captura de pantalla 2025-05-05 230257.png

## Ejecución

1. Asegurarse de tener Tesseract instalado.
2. Modificar la linea en "actividad1_filtrado_imagen.py" si la linea de Tesseract es diferente.
3. Ejecutar el script:
   python actividad1_filtrado_imagen.py
Se abrirán ventanas con las imágenes procesadas y se imprimirá el texto detectado en la terminal.

## Ejemplo de salida

Texto detectado en la placa: JKL
Texto detectado en la placa: 789JKL

## Autor

Paulina Méndez López
A01644629
