import cv2
import time
import numpy as np
import imutils
camera= 0
cam = cv2.VideoCapture(camera)
fgbg = cv2.createBackgroundSubtractorMOG2(history=1000,varThreshold=0,detectShadows=False)
width=600
height=480
fps_time = 0
while True:

    ret_val,image = cam.read()
    image = cv2.resize(image,(width,height))
    image = cv2.GaussianBlur(image, (5, 5), 0)
    fgmask = fgbg.apply(image)
    # image = fgbg.apply(image,learningRate=0.001)

    # image = imutils.resize(image, width=500)

    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    cnts = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]

	# loop over the contours
    x_left = -1
    y_left = -1
    x_right = -1
    y_right = -1
    for c in cnts:
		# if the contour is too small, ignore it
        # if cv2.contourArea(c) > 500:
        #     continue

		# compute the bounding box for the contour, draw it on the frame,
		# and update the text

        (x, y, w, h) = cv2.boundingRect(c)
        if x_left ==-1 :
            x_left = x
            y_left = y
        if x < x_left:
            x_left = x
        if y < y_left:
            y_left = y
        if x+w > x_right:
            x_right = x+w
        if y+h > y_right:
            y_right = y+h
        # cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    if (x_left==0 and y_left==0 and x_right==width and y_right==height)==False:
        cv2.rectangle(image, (x_left, y_left), (x_right, y_right), (0, 255, 0), 2)

    # cv2.putText(image,
    #                 "FPS: %f [press 'q'to quit]" % (1.0 / (time.time() - fps_time)),
    #                 (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
    #                 (0, 255, 0), 2)

    cv2.imshow('tf-pose-estimation result',fgmask)
    cv2.imshow('tf-pose-estimation result2',image)


    fps_time = time.time()
    if cv2.waitKey(1)==ord('q'):
        cam.release()
        cv2.destroyAllWindows()
        break
