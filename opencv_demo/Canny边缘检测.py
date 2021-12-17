import cv2.cv2 as cv2
from matplotlib import pyplot as plt

img = cv2.imread('./image/linyuner.jpeg', 0)
edges = cv2.Canny(img, 100, 200)

plt.subplot(121), plt.imshow(img, cmap='gray')
plt.title('original'), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(edges, cmap='gray')
plt.title('edge'), plt.xticks([]), plt.yticks([])

plt.show()
