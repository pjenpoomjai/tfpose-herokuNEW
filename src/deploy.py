#from flask import Flask
import paho.mqtt.client as mqtt
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# import processImage
import processImage_before
from processImage import Terrain
import os
import json

#app = Flask(__name__)

#@app.route('/')
def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))

    # Subscribing in on_connect() means that if we lose the connection and
    # reconnect then subscriptions will be renewed.
    client.subscribe("zenbo/image")
def on_message(client, userdata, msg):
    print ("Topic : ", msg.topic)
    # print(data["byteArr"])
    room = msg.payload[-1]
    image = msg.payload[:-1]
    f = open("./images/tet.jpg", "wb")  #there is a output.jpg which is different
    f.write(image)
    f.close()
    print('received image')
    processImage(room)
def run():
    broker_address = "broker.mqttdashboard.com"
    #broker_address = "iot.eclipse.org"
    client.on_connect = on_connect
    client.on_message = on_message  # attach function to callback
    print("connecting to broker")
    client.connect(broker_address)  # connect to broke
    client.loop_forever()
def processImage(room):
    nameImage = './images/tet.jpg'
    global round
    print('room : ',room,'---------begin mesh function.--------------',round)
    print('time : ',time.time())
    round = round + 1
    index = -1
    for i in range(len(rooms)):
        if room == rooms[i][0]:
            index = i
            break
    if index==-1:
        rooms = rooms + [room , Terrain()]
    try:
        t = rooms[index][1]
        t.mesh(nameImage)
        FALL_DETECTED = t.getBitFalling()
        print(room,', : Complete All')
        if FALL_DETECTED: #when found falling  turn FALL to True
            client.publish("FALL_DETECT", 'FALL_'+rooms[index][0])
    except Exception as e:
        print(e)
        print("Image not clear")
if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    rooms = []
    # t = Terrain()
    print("creating new instance")
    client = mqtt.Client('cloudPRocess')  # create new instance
    round = 1
    run()
