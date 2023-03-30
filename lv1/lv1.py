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
image = cv.imread('lv1/lv1_1.jpg')
image = resize(image,0.5)
original = image.copy()

# find contours
blur = cv.GaussianBlur(image,(3,3),0)
gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY|cv.THRESH_OTSU)[1]
contours, hierarchy =cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
print(len(contours))

# make different
n=0
dif1 = random.randint(0,len(contours))
dif2 = random.randint(0,len(contours))

for contour in contours:
    if(n==dif1 or n==dif2):
        n = n+1
        x, y, w, h = cv.boundingRect(contour)
        test = image[y:y+h,x:x+w].copy()
        image[y:y+h,x:x+w] = (191,146,112)
        height, width = test.shape[:2]
        rotation_matrix = cv.getRotationMatrix2D((width/2,height/2),random.randint(1,3)*90, 1)
        rotated_test = cv.warpAffine(test,rotation_matrix,(width,height))
        image[y:y+h,x:x+w] = rotated_test
    n=n+1

            
# correct background
mask = cv.inRange(image,(0,0,0),(100,100,100))
image[mask>0] = (191,146,112)

# save image to file
cv.imwrite('lv1/lv1_2.jpg',image)

# output image
blank = np.zeros((original.shape[0],30,3),dtype='uint8')
final = cv.hconcat([image, blank, original])
cv.imshow('Level 1',final)

cv.waitKey(0)
cv.destroyAllWindows()