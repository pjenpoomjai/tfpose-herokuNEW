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
    count_letter = int(msg.payload[-1])
    array_name_room = msg.payload[-(count_letter+1):-1]
    name_room = ""
    for assi in array_name_room:
        name_room = name_room + chr(int(assi))
    image = msg.payload[:-(count_letter+1)]
    f = open("./images/tet.jpg", "wb")  #there is a output.jpg which is different
    f.write(image)
    f.close()
    print('Received data.')
    print('Complete : split room number and image.',name_room)
    processImage(name_room)
def run():
    #broker_address = "broker.mqttdashboard.com"
    broker_address = "iot.eclipse.org"
    client.on_connect = on_connect
    client.on_message = on_message  # attach function to callback
    print("connecting to broker")
    client.connect(broker_address)  # connect to broke
    client.loop_forever()
def processImage(room):
    nameImage = './images/tet.jpg'
    global rounds
    global rooms
    global terrains
    index = -1
    for i in range(len(rooms)):
        if room == rooms[i]:
            index = i
            rounds[index] = rounds[index] + 1
            break
    if index==-1:
        rounds = rounds + [1]
        rooms = rooms + [room]
        terrains = terrains + [Terrain()]
        print("Create room number #",room)
    try:
        print('room : ',room,'----begin mesh function.-----',rounds[index])
        print('time : ',time.time())
        t = terrains[index]
        t.mesh(nameImage)
        FALL_DETECTED = t.getBitFalling()
        print('++',room,', : Complete mesh all++')
        if FALL_DETECTED: #when found falling  turn FALL to True
            word = 'FALL_'+room
            f = open("./images/tet.jpg", "rb")  #there is a output.jpg which is different
            image = f.read()
            f.close()
            client.publish("zenbofall/linebot", image)
            client.publish("FALL_DETECT", word)
            print('Send signal to zenbo. ( ' +word+ " )")
            print('Send image too.')
            print('.')
            print('. .')
            print('Complete.')
    except Exception as e:
        print(e)
        print("Image not clear")
if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    rooms = []
    terrains = []
    # t = Terrain()
    print("creating new instance")
    client = mqtt.Client('cloudPRocess')  # create new instance
    rounds = []
    run()
