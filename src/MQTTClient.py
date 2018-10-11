import paho.mqtt.client as mqtt  # import the client1
import time
############
import cv2

from estimator import TfPoseEstimator
from networks import get_graph_path, model_wh

broker_address = "iot.eclipse.org"
print("creating new instance")
client = mqtt.Client("client1")  # create new instance
# client.on_message = on_message  # attach function to callback

print("connecting to broker")
client.connect(broker_address)  # connect to broker
#print("Publishing message to topic", "if/test")
message = 'end'

camera = 0
recordTime =0
f = cv2.VideoCapture(camera)
numberCount = 0
listNameImage = range(100) #when save a image from camera

# model = 'mobilenet_thin_432x368'
# w, h = model_wh(model)
model = 'cmu'
w, h = 656, 368
e = TfPoseEstimator(get_graph_path(model), target_size=(w, h))


while True:
    ret_int,img = f.read()
    #picName = time.asctime( time.localtime(time.time())).replace(':','_')
    #picName = picName.replace(' ','_') +".jpg"

    cv2.imshow('came',img)


    humans = e.inference(img, scales=[None])
    package = TfPoseEstimator.draw_humans(img, humans, imgcopy=False)
    img = package[0]


    #if recordTime!=int(time.time()):    3 picture / sec
    if time.time() - recordTime > 0.3:
        picName = str(listNameImage[numberCount])+'.jpg'
        numberCount = numberCount + 1
        if numberCount >= len(listNameImage):
            numberCount = 0
        print(picName)
        cv2.imwrite(picName, img)
        recordTime = time.time()
        fileImage = open(picName,'rb')
        fileImage = fileImage.read()
        byteArr = bytearray(fileImage)
        print(time.time())
        print("Publishing message to topic", "zenbo/image")
        client.publish(topic="zenbo/image", payload= byteArr ,qos=0)
        #ledStatus 3 open light 2close light ,FLUKE
        #client.publish(topic="ledStatus", payload= '2' ,qos=0)
        print('Complete')

    if cv2.waitKey(1)==ord('q'):
        f.release()
        cv2.destroyAllWindows()
        break
#f = open('a.jpg','rb')
#fileImage = f.read()
#f.close()
#byteArr = bytearray(fileImage)
#print(byteArr)
#client.publish("zenbo/message", message)
#client.publish("zenbo/message", byteArr,0)
