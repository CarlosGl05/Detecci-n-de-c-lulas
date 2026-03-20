import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("blood.jpeg")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Aplicar filtro de Sobel para detectar bordes
grad_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
grad_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)

mag = cv2.magnitude(grad_x, grad_y)
mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

mag_rgb = cv2.cvtColor(mag, cv2.COLOR_GRAY2RGB)

_, mask = cv2.threshold(mag, 50, 255, cv2.THRESH_BINARY)

# Kernel con forma de elipse
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
# Operacion morfologica para erosionar y luego dilatar (Eliminar ruido)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
# Operacion morfologica para dilatar y luego erosionar (Rellenar huecos negros)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

# Desenfoque para eliminar ruido 
gray_blurred = cv2.medianBlur(img_gray, 5)

circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, dp = 1.2, minDist=25, param1=50, param2=21, minRadius=20, maxRadius=55)

if circles is not None:
  circles = np.uint16(np.around(circles))
  for i in circles[0, :]:
      # Dibuja el contorno del círculo en rojo
      cv2.circle(img, (i[0], i[1]), i[2], (0, 0, 255), 2)
      cv2.circle(img, (i[0], i[1]), 2, (0, 255, 0), 3)
  
  num_circles = len(circles[0])
else:
  num_circles = 0

print(f'Células detectadas: {num_circles}')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

fig, axs = plt.subplots(2, 3, figsize=(10, 8))
axs[0, 0].imshow(grad_x, cmap='gray')
axs[0, 0].set_title("Gradiente X")
axs[0, 0].axis("off")

axs[0, 1].imshow(grad_y, cmap='gray')
axs[0, 1].set_title("Gradiente Y")
axs[0, 1].axis("off")

axs[0, 2].imshow(mag, cmap='gray')
axs[0, 2].set_title("Magnitud")
axs[0, 2].axis("off")

axs[1, 0].imshow(mask, cmap='gray')
axs[1, 0].set_title("Threshold")
axs[1, 0].axis("off")

axs[1, 1].imshow(gray_blurred, cmap='gray')
axs[1, 1].set_title("Blurred")
axs[1, 1].axis("off")

axs[1, 2].imshow(img)
axs[1, 2].set_title("Circles")
axs[1, 2].axis("off")

plt.tight_layout()
plt.show()
