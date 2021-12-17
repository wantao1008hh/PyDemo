import cv2.cv2 as cv2

for i in dir(cv2):
       if i.startswith('COLOR_'):
           print(i)
