import cv2
import numpy as np
import matplotlib.pyplot as plt

inp = cv2.imread("input.jpeg")
inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(inp, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)

grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
grad = np.sqrt(grad_x**2, grad_y**2)
grad_norm = (255 * grad / grad.max())




plt.subplot(2, 3, 1)
plt.imshow(inp)
plt.axis('off')
plt.title('Input')

plt.subplot(2, 3, 2)
plt.imshow(gray, cmap = 'gray')
plt.axis('off')
plt.title('Gray Scale')

plt.subplot(2, 3, 3)
plt.imshow(blur, cmap = 'gray')
plt.axis('off')
plt.title('Gaussian Blur')

plt.subplot(2, 3, 4)
plt.imshow(grad_norm, cmap = 'gray')
plt.axis('off')
plt.title('Edges')

plt.show()
