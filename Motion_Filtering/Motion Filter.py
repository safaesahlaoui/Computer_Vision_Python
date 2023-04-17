import  cv2 as cv
cap=cv.VideoCapture("C:/Users/hp/Desktop/vid.mp4")
substractor = cv.createBackgroundSubtractorMOG2(30,50)
while True:
    ret,frame = cap.read()
    if ret :
        mask = substractor.apply(frame)
        cv.imshow('Mask', mask)

        if cv.waitKey(5)==ord('x'):
            break
    else:
        cap=cv.VideoCapture("C:/Users/hp/Desktop/vid.mp4")

cv.destroyAllWindows()
cap.release()