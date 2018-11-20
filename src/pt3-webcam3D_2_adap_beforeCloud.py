
import sys
import cv2
import time
#import os
import paho.mqtt.client as mqtt
from estimator import TfPoseEstimator
from networks import get_graph_path, model_wh
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import imutils


class Terrain(object):

    def __init__(self):
        """
        Initialize the graphics window and mesh surface
        """
        self.bitFalling = 0
        # Initialize plot.
        plt.ion()
        f2 = plt.figure(figsize=(6, 5))
        self.windowNeck = f2.add_subplot(1, 1, 1)
        self.windowNeck.set_title('Speed')
        self.windowNeck.set_xlabel('Time')
        self.windowNeck.set_ylabel('Speed')

        # plt.show()
        self.times = []
        self.recordVelocity = [0]
        self.recordNeck = []
        self.recordYTopRectangle = []
        self.recordHIP = []
        self.recordNeck_Rshoulder = []
        self.recordTimeList = []
        self.globalTime = 0
        self.fps_time = 0
        self.highestNeck = 0
        self.recordTimeNeckHighest = 0
        self.highestHIP = 0
        self.saveTimesStartFalling = -1

        self.recordTimeHIPHighest = 0
        self.surpriseMovingTime = -1
        self.detectedHIP_Y = 0
        self.detectedNECK_Y = 0
        self.extraDistance = 0
        #add more than adapt
        self.fgbg = cv2.createBackgroundSubtractorMOG2(history=1,varThreshold=100,detectShadows=False)
        self.secondNeck = 0
        self.human_in_frame = False
        self.lastTimesFoundNeck = -1
        self.width = 300
        self.height = 300
        self.quotaVirtureNeck = 1
        self.used_quotaVirtureNeck = 0
        model = 'mobilenet_thin_432x368'
        w, h = model_wh(model)
        #model = 'cmu'
        #w, h = 656, 368
        camera = 0  # 1 mean external camera , 0 mean internal camera
        self.e = TfPoseEstimator(get_graph_path(model), target_size=(w, h))
        self.cam = cv2.VideoCapture(camera)
        # self.cam.set(10,255)
        # self.cam.set(11, 1   ) # contrast       min: 0   , max: 255 , increment:1
        # self.cam.set(12,17)
        ret_val, image = self.cam.read(cv2.IMREAD_GRAYSCALE)
        try:
            self.mesh(image)
        except Exception as e:
            pass
            # print(e)
    def reduceRecord(self) :
        self.recordNeck = self.recordNeck[-100:]
        self.recordHIP = self.recordHIP[-100:]
        self.times = self.times[-100:]
        self.recordVelocity = self.recordVelocity[-100:]
        self.recordTimeList = self.recordTimeList[-100:]
        self.recordNeck_Rshoulder = self.recordNeck_Rshoulder[-100:]
        self.recordYTopRectangle = self.recordYTopRectangle[-100:]
    def getLastRecordTime(self):
        if self.recordTimeList==[]:
            return 0
        return self.recordTimeList[-1]
    def addCountTimes(self):
        if self.times == []:
            self.times = self.times + [1]
        else:
            self.times = self.times + [self.times[-1]+1]
    def addRecordTime(self,time):
        self.recordTimeList = self.recordTimeList + [time]
    def addRecordHIP(self,hip):
        self.recordHIP = self.recordHIP + [hip]
    def addRecordNeck(self,neck):
        self.recordNeck = self.recordNeck + [neck]
    def addRecordVelocity(self,neck,time):
        v = ( abs(neck[-1] - neck[-2]) / abs(time[-1] - time[-2]) )
        self.recordVelocity = self.recordVelocity + [int(v)]
    def addRecordNeck_RShoulder(self,length):
        self.recordNeck_Rshoulder = self.recordNeck_Rshoulder+[length]
    def getLengthBetweenPoint(self,pointA,pointB):
        x = (pointA[0] - pointB[0])**2
        y = (pointA[1] - pointB[1])**2
        return (abs(x - y)**(1/2))
    def destroyAll(self):
        self.times = []
        self.recordNeck = []
        self.recordHIP = []
        self.recordTimeList = []
        self.recordNeck_Rshoulder = []
        self.recordVelocity = [0]
        self.recordYTopRectangle = []
        self.resetSurpriseMovingTime()
        self.resetBitFalling()
    def addFPStoWindow(self,window,timeSave):
        cv2.putText(window,
                    "FPS: %f [press 'q'to quit]" % (
                        1.0 / (timeSave - self.fps_time)),
                    (10, 20),  cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2)
    def detecedFirstFalling(self):
        self.detectedNECK_Y = self.highestNeck
        self.detectedHIP_Y  = self.highestHIP
        print('-------------------------------!!!!falling!!!!!!-----------------')
        print('-------------------------------!!!!falling!!!!!!-----------------')

        # print('HIGHEST NECK',self.highestNeck)
        # print('current NECK',self.getLastNeck())
        # print('result [ neck ]current - HIGHEST: ',abs(self.getLastNeck() - self.highestNeck))
        self.surpriseMovingTime = self.globalTime
        self.saveTimesStartFalling = self.times[-1]
        #low value then far from camera
        # print('set extraDistance')
        # print(min(self.recordNeck_Rshoulder[-7:-2]))
        # print(self.recordNeck_Rshoulder[-1])
        # minNeckRShoulder = min(self.recordNeck_Rshoulder[-7:-2])
        # if self.recordNeck_Rshoulder[-1] > minNeckRShoulder:
        #     print('ENTER CAMERA')
        #     self.extraDistance = (self.detectedHIP_Y - self.detectedNECK_Y)
        #
        # else:
        #     print('OUT CAMERA')
        #     rate = self.recordNeck_Rshoulder[-1]/minNeckRShoulder
        #     self.extraDistance = rate*(self.detectedHIP_Y - self.detectedNECK_Y)
        #     print(rate)
        #
        self.extraDistance = (self.detectedHIP_Y - self.detectedNECK_Y)*(1/2)
        # print('extraDis : ',self.extraDistance)
        # print('set complete ')
    def countdownFalling(self):
        # print('----------------------------------------')
        # print('StartTime From: ',self.surpriseMovingTime)
        print('!!!!!Countdown[10] : ',self.globalTime - self.surpriseMovingTime,'!!!!!')
        # print('would like to Cancel Countdown \nTake your neck to same level as NECK , HIP : ',self.detectedNECK_Y,self.detectedHIP_Y)
        # print('current your NECK : ',self.getLastNeck())
        # print('extraTotal:',self.detectedHIP_Y+self.extraDistance)
        print('----------------------------------------')
        #maybe not Falling but make sure with NECK last must move up to this position
        # print('check STATE 2')
    def resetSurpriseMovingTime(self):
        self.surpriseMovingTime=-1
    def getLastNeck(self):
        return self.recordNeck[-1]
    def getLastTimes(self):
        return self.times[-1]
    def getSecondNeck(self):
        return self.secondNeck
    def getLastTimesFoundNeck(self):
        return self.lastTimesFoundNeck
    def addStatusFall(self,image):
        color = (0, 255, 0)
        if self.surpriseMovingTime!=-1:
            color = (0, 0, 255)
        cv2.circle(image,(10,10), 10, color, -1)
    def savesecondNeck(self,image):
        blur = cv2.GaussianBlur(image, (5, 5), 0)
        fgmask = self.fgbg.apply(blur)
        cnts = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL,
    		cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        x_left = -1
        y_left = -1
        x_right = -1
        y_right = -1
        for c in cnts:
    		# if the contour is too small, ignore it
            # if cv2.contourArea(c) > 500:
            #     continue
    		# compute the bounding box for the contour, draw it on the frame,
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
        if (x_left==0 and y_left==0 and x_right==self.width and y_right==self.height)==False:
            cv2.rectangle(image, (x_left, y_left), (x_right, y_right), (0, 255, 0), 2)
            if self.human_in_frame and y_left != -1:
                self.secondNeck = y_left
                print('second Neck : ',self.secondNeck)
                self.recordYTopRectangle = self.recordYTopRectangle + [self.secondNeck]
        cv2.imshow('na',fgmask)
    def mesh(self, image):
        # print('start-inderence',time.time())
        humans = self.e.inference(image, scales=[None])
        # print('end-inderence',time.time())
        package = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
        self.globalTime = time.time()  #time of after drawing
        # print(self.globalTime)
        image = package[0]
        status_part_body_appear = package[1]
        center_each_body_part = package[2]
        self.resetBitFalling()
        self.savesecondNeck(image)
        print('insert FPS')
        timeSave = time.time()
        if timeSave - self.fps_time > 0:
            self.addFPStoWindow(image,timeSave)
        print('show image')
        self.addStatusFall(image)
        cv2.imshow('tf-pose-estimation result2', image)
        # self.fps_time = time.time()
        #camera not found NECK more than 10 second then reset list
        if self.globalTime - self.getLastRecordTime() >= 12:
            # print('RESET STABLE,RECORDNECK,HIP,etc. [complete 12 second]')
            self.destroyAll()
        if self.globalTime - self.getLastRecordTime() >= 2:
            # print('maybe NECK or HUMAN not found [complete 2 second]')
            self.human_in_frame=False
        # print('end Initialize mesh')
        #find length of neck , R_SHOULDER
        # if 2 in center_each_body_part and 1 in center_each_body_part:
        #     p1 = center_each_body_part[2]
        #     p2 = center_each_body_part[1]
        #     self.addRecordNeck_RShoulder(self.getLengthBetweenPoint(p1,p2))
        # if 2 in center_each_body_part and 1 in center_each_body_part:
        #     #R_SHOULDER X point                   Neck X point
        #     if center_each_body_part[2][0] < center_each_body_part[1][0]:
        #         print('Front')
        #     elif center_each_body_part[2][0] > center_each_body_part[1][0]:
        #         print('Back')

        # print(status_part_body_appear)
        #when draw2D stick man
        # name_part_body = ["Nose",  # 0
        #                   "Neck",  # 1
        #                   "RShoulder",  # 2
        #                   "RElbow",  # 3
        #                   "RWrist",  # 4
        #                   "LShoulder",  # 5
        #                   "LElbow",  # 6
        #                   "LWrist",  # 7
        #                   "RHip",  # 8
        #                   "RKnee",  # 9
        #                   "RAnkle",  # 10
        #                   "LHip",  # 11
        #                   "LKnee",  # 12
        #                   "LAnkle",  # 13
        #                   "REye",  # 14
        #                   "LEye",  # 15
        #                   "REar",  # 16
        #                   "LEar",  # 17
        #                   ]
        # detected_part = []
        #UPDATE highest y point NECK  every 1
        # print('start record everything')
        #mean not found neck in this frame
        if self.globalTime - self.getLastRecordTime() >= 0.25 :  # every 0.3 second record
            if 1 in center_each_body_part:
                # print(self.globalTime - self.getLastRecordTime())
                self.addCountTimes()
                self.addRecordTime(self.globalTime)
                self.human_in_frame = True
                self.lastTimesFoundNeck =self.getLastTimes()
                self.used_quotaVirtureNeck=0
                self.addRecordNeck(center_each_body_part[1][1])
                self.addRecordVelocity(self.recordNeck,self.recordTimeList)
                if 11 in center_each_body_part:
                    self.addRecordHIP(center_each_body_part[11][1])
                    print('neck :| HIP: ',self.recordHIP[-1] - self.recordNeck[-1])
                elif 8 in center_each_body_part:
                    self.addRecordHIP(center_each_body_part[8][1])
                    print('neck :| HIP: ',self.recordHIP[-1] - self.recordNeck[-1])
            elif self.getLastTimesFoundNeck()==self.getLastTimes() and self.used_quotaVirtureNeck<=self.quotaVirtureNeck:
                # print(self.globalTime - self.getLastRecordTime())
                self.addCountTimes()
                self.addRecordTime(self.globalTime)
                self.lastTimesFoundNeck =self.getLastTimes()
                self.addRecordNeck(self.getSecondNeck())
                self.addRecordVelocity(self.recordYTopRectangle,self.recordTimeList)
                # print('addSecond Neck')
                self.used_quotaVirtureNeck+=1
            if len(self.recordNeck) > 600: #when record list more than 600 -> reduce
                self.reduceRecord()
        # print('find highest neck , hip')
        if len(self.recordNeck)>1:
            if (self.getLastNeck() < self.highestNeck) or (self.globalTime - self.recordTimeNeckHighest >= 0.25):
                self.recordTimeNeckHighest = self.globalTime
                #found last 6  min value
                # print('find index')
                self.highestNeck = min(self.recordNeck[-6:]) #more HIGH more low value
                if len(self.recordHIP)>1:
                    #11 L_HIP
                    if 11 in center_each_body_part:
                        if center_each_body_part[11][1] < self.highestHIP :
                            self.highestHIP = center_each_body_part[11][1]
                            self.recordTimeHIPHighest = self.globalTime
                        elif self.globalTime - self.recordTimeHIPHighest >= 0.25:
                            #self.highestHIP = self.recordHIP[minIndex]
                            self.highestHIP = min(self.recordHIP[-6:])
                            self.recordTimeHIPHighest = self.globalTime
                    #8 R_HIP
                    elif 8 in center_each_body_part:
                        if center_each_body_part[8][1] < self.highestHIP :
                            self.highestHIP = center_each_body_part[8][1]
                            self.recordTimeHIPHighest = self.globalTime
                        elif self.globalTime - self.recordTimeHIPHighest >= 0.25:
                            #self.highestHIP = self.recordHIP[minIndex]
                            self.highestHIP = min(self.recordHIP[-6:])
                            self.recordTimeHIPHighest = self.globalTime
        # found NECK
        # print('processing falling ---------')
        # print('NECK : -',self.recordNeck)
        # print('HIP : -',self.recordHIP)
        if self.highestHIP!=0 and len(self.recordNeck)>1 and self.surpriseMovingTime==-1:
            #NECK new Y point > NECK lastest Y point      falling
            #high , y low     || low , y high
            # print('result [ neck ]current - HIGHEST: ',abs(self.getLastNeck() - self.highestNeck))
            # print('Top NECk ',self.highestNeck,'  Last Neck ',self.getLastNeck())
            # <100 walk , sit ground , pick up something
            # >100 suddently fall or suddently action
            h = [0,50,75,105]
            v = [80,100 , 150 , 250]
            for i in range(len(h)):
                if self.highestHIP - self.highestNeck>=h[i]:
                    velocity = v[i]
            print('velocity ', velocity)
            print('person Velocity', self.recordVelocity[-1])
            if self.recordVelocity[-1] > velocity:
                if (self.getLastNeck() > self.highestNeck) and (self.getLastNeck() > self.highestHIP ):
                    self.detecedFirstFalling()
        elif self.surpriseMovingTime!=-1:
            self.countdownFalling()
            # print('times - times : ',self.times[-1] - self.saveTimesStartFalling)
            if self.globalTime - self.surpriseMovingTime >= 2 and (self.getLastNeck() <= (self.detectedHIP_Y - self.extraDistance)):
                print('NECK : ',self.recordNeck)
                print('REC :',self.recordYTopRectangle)
                print('Is neck < recover ',self.getLastNeck()  , (self.detectedHIP_Y - self.extraDistance))
                print('---------------------------------------')
                print('Recover From STATE')
                print('---------------------------------------')
                self.destroyAll()
            elif self.globalTime - self.surpriseMovingTime >= 10:
                self.setFalling()
                print("Publishing message to topic", "zenbo/messageFALL")
                client.publish("zenbo/messageFALL", 'FALL DETECTED')
                self.destroyAll()
        # print('end processing falling end mash()')
    def setFalling(self):
        self.bitFalling = 1
    def getBitFalling(self):
        return self.bitFalling
    def resetBitFalling(self):
        self.bitFalling = 0
    def update(self):
        """
        update the mesh and shift the noise each time
        """
        ret_val, image = self.cam.read()
        try:
            # print('NEWROUND')
            image = cv2.resize(image, (self.width, self.height))
            cv2.imshow('normal', image)
            self.mesh(image)
            # print('--generateGraphStable--')
            # self.generateGraphStable()
            # print('COMPLETE-')
        except Exception as e:
            print('ERROR : -> ',e)
            pass
            #print('body not in image')
    def generateGraphStable(self):
        plt.cla()
        self.windowNeck.set_ylim([0, 300])
        plt.yticks(range(0, 300, 20), fontsize=14)
        plt.xlim(0,600)
        plt.plot(self.times, self.recordVelocity)
        # print('--- Times : ',self.getLastTimes(),'||| plot at Time : ',self.getLastRecordTime(),'||| Value : ',self.getLastNeck())
        plt.pause(0.01)
        # print('finish')
    def animation(self):
        while True:
            self.update()
            if cv2.waitKey(1) == ord('q'):
                self.cam.release()
                cv2.destroyAllWindows()
                break
if __name__ == '__main__':
    # os.chdir('..')
    t = Terrain()
    broker_address = "broker.mqttdashboard.com"
    #broker_address = "iot.eclipse.org"
    # print("creating new instance")
    client = mqtt.Client("comProcess")  # create new instance
    # client.on_message = on_message  # attach function to callback
    # print("connecting to broker")
    client.connect(broker_address)  # connect to broker
    t.animation()
