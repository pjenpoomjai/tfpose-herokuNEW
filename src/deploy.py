#from flask import Flask
import paho.mqtt.client as mqtt
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import processImage
from processImage import Terrain
import os

#app = Flask(__name__)

#@app.route('/')
def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))
        
    # Subscribing in on_connect() means that if we lose the connection and
    # reconnect then subscriptions will be renewed.
    client.subscribe("zenbo/image")
def on_message(client, userdata, msg):
    print ("Topic : ", msg.topic)
    f = open("./images/tet.jpg", "wb")  #there is a output.jpg which is different
    f.write(msg.payload)
    f.close()
    print('received image')
    processImage()
def run():
    broker_address = "iot.eclipse.org"
    client.on_connect = on_connect
    client.on_message = on_message  # attach function to callback
    print("connecting to broker")
    client.connect(broker_address)  # connect to broke
    client.loop_forever()
def processImage():
    nameImage = './images/tet.jpg'
    print('begin mesh function.')
    try:
        t.mesh(nameImage)
        print('Show 3D image')
        #t.show3DImage()
        print(t.get_graph_stable())
        print('Complete All')
        client.publish("zenbo/messageFALL", 'FALL DETECTED')
    except:
        print("Image not clear")
if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    t = Terrain()
    print("creating new instance")
    client = mqtt.Client('cloudPRocess')  # create new instance
    run()
