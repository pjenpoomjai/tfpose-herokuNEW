import paho.mqtt.client as mqtt  # import the client1
import time
import argparse
############
import cv2
# broker.mqttdashboard.com

parser = argparse.ArgumentParser(description='Sent Image to cloud')
parser.add_argument('--room', default='1', help='number room')
args = parser.parse_args()

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
round = 1
# client.publish(topic="nonine", payload= "FALL" ,qos=0)
while True:
    ret_int,img = f.read()
    cv2.imshow('came',img)
    #if recordTime!=int(time.time()):    3 picture / sec
    if time.time() - recordTime > 0.3:
        pathName = './images/'
        picName = pathName+str(listNameImage[numberCount])+'.jpg'
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
        client.publish(topic="zenbo/image", payload= [byteArr,args.room] ,qos=0)
        print(args.room,',Complete : ',round)
        round = round + 1
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
