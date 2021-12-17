import cv2.cv2 as cv2

img = cv2.imread('./image/linyuner.jpeg')

cv2.namedWindow("linyuner", cv2.WINDOW_AUTOSIZE)
print(img.shape)
#cv2.split()是比较耗时的操作，能用numpy就尽量使用。
print(img[:, :, 0])

cv2.imshow("linyuner", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# px = img[100, 100]
# print(px)
# blue = img[100, 100, 0]
# print(blue)
