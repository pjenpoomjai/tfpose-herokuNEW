import cv2
import time
camera= 1
cam = cv2.VideoCapture(camera)
width=800
height=640
fps_time = 0
while True:
    
    ret_val,image = cam.read()

    
    cv2.putText(image,
                    "FPS: %f [press 'q'to quit]" % (1.0 / (time.time() - fps_time)),
                    (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)
    image = cv2.resize(image,(width,height))
    cv2.imshow('tf-pose-estimation result',image)
    fps_time = time.time()
    if cv2.waitKey(1)==ord('q'):
        cam.release()
        cv2.destroyAllWindows()
        break
