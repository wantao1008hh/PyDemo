import cv2.cv2 as cv2
import numpy as np
img1=cv2.imread('./image/linyuner.jpeg')
print(img1.shape)

cv2.imshow("原始", img1)
(h, w) = img1.shape[:2]
(cX, cY) = (w // 2, h // 2)

M = cv2.getRotationMatrix2D((cX, cY), -90, 1.0)
rotated = cv2.warpAffine(img1, M, (w, h))
print(rotated.shape)


cv2.imshow("Rotated by -90 Degrees", rotated)



dst = cv2.addWeighted(img1,0.7,rotated,0.3,0)
cv2.imshow('dst',dst)
cv2.waitKey(0)
cv2.destroyAllWindows()