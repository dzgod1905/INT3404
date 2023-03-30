import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import random

def resize(image, scale = 0.75):
    width = int(image.shape[1] * scale)
    height = int(image.shape[0] * scale)
    dimensions = (width,height)

    return cv.resize(image, dimensions, interpolation=cv.INTER_CUBIC)

# read image
image = cv.imread('lv2/lv2_1.jpg')
image = resize(image,0.5)
original = image.copy()

# find contours
blur = cv.GaussianBlur(image,(3,3),0)
gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY|cv.THRESH_OTSU)[1]
contours, hierarchy =cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

n=0
dif1 = random.randint(0,len(contours))
dif2 = random.randint(0,len(contours))
dif3 = random.randint(0,len(contours))
dif4 = random.randint(0,len(contours))

for contour in contours:
    if(n==dif1 or n==dif2 or n==dif3 or n==dif4):
        x, y, w, h = cv.boundingRect(contour)
        test = image[y:y+h,x:x+w].copy()
        mask_red = cv.inRange(test,(36-36,27-27,237-50),(36+50,27+50,255))
        mask_yellow = cv.inRange(test,(0,242-50,254-50),(0+50,255,255))
        mask_pink = cv.inRange(test,(201-50,174-50,254-50),(201+50,174+50,254+1))
        mask_green = cv.inRange(test,(77-50,177-50,35-35),(77+50,177+50,35+50))
        mask_orange = cv.inRange(test,(38-38,127-50,255-50),(38+50,127+50,255))
        mask_blue = cv.inRange(test,(232-50,163-50,0),(255,163+50,0+50))
        test[mask_red>0] = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
        test[mask_yellow>0] = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
        test[mask_pink>0] = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
        test[mask_green>0] = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
        test[mask_orange>0] = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
        test[mask_blue>0] = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
        image[y:y+h,x:x+w] = test
    n=n+1

# save image
cv.imwrite('lv2/lv2_2.jpg',image)

# output image
blank = np.zeros((original.shape[0],30,3),dtype='uint8')
final = cv.hconcat([image, blank, original])
cv.imshow('Level 2',final)

cv.waitKey(0)
cv.destroyAllWindows()