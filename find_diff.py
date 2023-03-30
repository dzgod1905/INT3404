import cv2 as cv
import imutils
import numpy as np

# select the level to find diff
lvl = 'lv2'

def resize(frame, scale = 0.75):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width,height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_CUBIC)

# load and resize the image
img1 = cv.imread(lvl+'/'+lvl+'_1.jpg')
img1 = resize(img1, 0.5)
img2 = cv.imread(lvl+'/'+lvl+'_2.jpg')

# convert to grayscale and find diff
img1Gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
img2Gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
diff = cv.absdiff(img1Gray,img2Gray)

# find contours
threshold = cv.threshold(diff,0,255,cv.THRESH_BINARY|cv.THRESH_OTSU)[1]
kernel = np.ones((3,3),np.uint8)
dilate = cv.dilate(threshold, kernel, iterations=2)
contours = cv.findContours(dilate.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)

# loop over each contours to draw a box
for contour in contours:
    if(cv.contourArea(contour) > 0):
        x, y, w, h = cv.boundingRect(contour)
        cv.rectangle(img1, (x,y), (x+w,y+h), (0,0,255), 2)
        cv.rectangle(img2, (x,y), (x+w,y+h), (0,0,255), 2)

# final img
blank = np.zeros((img1.shape[0],20,3),dtype='uint8')
final = cv.hconcat([img1, blank, img2])
cv.imshow(lvl,final)

cv.waitKey(0)
cv.destroyAllWindows()