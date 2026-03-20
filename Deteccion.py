import cv2
import numpy as np
import matplotlib.pyplot as plt

def deteccion_con_hough(img):
    # convierte a gris
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # --- Conservamos esto solo para la visualización de los pasos ---
    # cálculo de gradiente sobel (horizontal + vertical)
    grad_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)

    mag = cv2.magnitude(grad_x, grad_y)
    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # umbral y limpieza de máscara
    _, mask = cv2.threshold(mag, 20, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    # ---------------------------------------------------------------

    # --- APLICACIÓN DE HOUGH CIRCLES ---
    # Suavizamos la imagen en gris para evitar que el ruido genere falsos círculos
    gray_blurred = cv2.medianBlur(img_gray, 5)

    # Detectar círculos
    circles = cv2.HoughCircles(
        gray_blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,           # Aumentar ligeramente la resolución del acumulador ayuda con círculos difusos
        minDist=30,       # Reducimos un poco la distancia mínima entre centros
        param1=40,        # ¡CLAVE! Baja el umbral del detector de bordes Canny (antes 50). Así detecta bordes suaves.
        param2=25,        # ¡CLAVE! Baja la exigencia de "perfección". Un valor más bajo (antes 25) detectará más círculos, aunque no sean perfectos.
        minRadius=20,     # Asegúrate de que este valor sea un poco más pequeño que tu glóbulo rojo más pequeño
        maxRadius=70
    )

    output = img.copy()
    num_circles = 0

    # Dibujar los círculos si se encontraron
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # dibuja el contorno del círculo en rojo
            cv2.circle(output, (i[0], i[1]), i[2], (0, 0, 255), 2)
            # Opcional: dibuja el centro del círculo
            # cv2.circle(output, (i[0], i[1]), 2, (0, 255, 0), 3)
        
        num_circles = len(circles[0])

    return img_gray, mag, mask, output, num_circles

def main():
    img = cv2.imread('celula.jpeg')
    if img is None:
        raise FileNotFoundError('No se encontró la imagen celula.jpeg')

    img_gray, mag, mask, output, num_circles = deteccion_con_hough(img)

    print(f'Células (círculos) detectadas: {num_circles}')

    # Mostrar resultados
    titles = ['Gris', 'Magnitud Gradiente', 'Mascara Sensible', 'Círculos Detectados (Hough)']
    images = [img_gray, mag, mask, cv2.cvtColor(output, cv2.COLOR_BGR2RGB)]

    plt.figure(figsize=(12, 8))
    for i, (title, im) in enumerate(zip(titles, images), start=1):
        plt.subplot(2, 2, i)
        if len(im.shape) == 2:
            plt.imshow(im, cmap='gray')
        else:
            plt.imshow(im)
        plt.title(title)
        plt.axis('off')

    plt.tight_layout()
    plt.show()

main()