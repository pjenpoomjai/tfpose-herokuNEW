from flask import Flask
import paho.mqtt.client as mqtt
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from src import processImage
app = Flask(__name__)

@app.route('/picture')
def picture():

    img=mpimg.imread('./images/tet.jpg')
    imgplot = plt.imshow(img)
    plt.show()


@app.route('/home')
def home():
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
        client.subscribe("zenbo/message")
    def on_message(client, userdata, msg):
        print ("Topic : ", msg.topic)
        f = open("./images/tet.jpg", "wb")  #there is a output.jpg which is different
        f.write(msg.payload)
        f.close()
        picture()
        processImage()
        client.loop_stop()



    run = True
    broker_address = "iot.eclipse.org"
    print("creating new instance")

    client = mqtt.Client()  # create new instance
    client.on_connect = on_connect

    client.on_message = on_message  # attach function to callback

    print("connecting to broker")
    client.connect(broker_address)  # connect to broke
    client.loop_forever()
def processImage():
    t = Terrain()
    try:
        nameImage = './images/tet.jpg'
        t.mesh(nameImage)
        t.show_3D_image()
    except:
        pass
if __name__ == "__main__":
    app.run()
