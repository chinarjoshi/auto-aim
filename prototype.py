import numpy as np
import cv2
import matplotlib.pyplot as plt


image = cv2.imread('cup.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cup_cascade = cv2.CascadeClassifier('model.xml')
cup = cup_cascade.detectMultiScale(gray, 50, 50)

for x, y, w, h in cup:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)


cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
