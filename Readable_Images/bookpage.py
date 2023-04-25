import cv2 as cv

img = cv.imread('E:\Safae\Pycharm Projects\Computer Vision Python\Images/book.jpg')
img = cv.cvtColor(img,cv.COLOR_RGB2GRAY )
_,result=cv.threshold(img,100,255,cv.THRESH_BINARY)
adaptative = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,21,2)
cv.imshow('Image',img)
cv.imshow('Result',result)
cv.imshow('Adaptive Result ',adaptative)
cv.waitKey(0)
cv.destroyAllWindows()
