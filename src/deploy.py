from flask import Flask
import paho.mqtt.client as mqtt
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import processImage
from processImage import Terrain
app = Flask(__name__)

@app.route('/picture')
def picture():

    img=mpimg.imread('./images/tet.jpg')
    imgplot = plt.imshow(img)
    plt.show()


@app.route('/')
def index():
    # def on_message(client, userdata, message):
    #     print("message recgeived ", str(message.payload.decode("utf-8", "ignore")))
    #     print("message topic=", message.topic)
    #     print("message qos=", message.qos)
    #     print("message retain flag=", message.retain)
    #     if (str(message.payload.decode("utf-8", "ignore")) == "end"):
    #         global run
    #         run = False
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
        #picture()
        processImage()



    run = True
    broker_address = "iot.eclipse.org"
    client.on_connect = on_connect

    client.on_message = on_message  # attach function to callback

    print("connecting to broker")
    client.connect(broker_address)  # connect to broke
    client.loop_forever()
def processImage():
    # print('create object Terrain')


    nameImage = './images/tet.jpg'
    print('begin mesh function.')
    #client.publish("zenbo/messageFALL", 'FALL DETECTED')
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
    t = Terrain()
    print("creating new instance")
    client = mqtt.Client('client')  # create new instance
    app.run()
