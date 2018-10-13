
import sys
import cv2
import time
#import os

from estimator import TfPoseEstimator
from networks import get_graph_path, model_wh
from matplotlib import style
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3


class Terrain(object):

    def __init__(self):
        """
        Initialize the graphics window and mesh surface
        """
        # Initialize plot.
        self.bitFalling = 0
        plt.ion()
        # f = plt.figure(figsize=(5, 5))
        f2 = plt.figure(figsize=(6, 5))

        self.windowNeck = f2.add_subplot(1, 1, 1)
        self.windowNeck.set_title('Stable')
        self.windowNeck.set_xlabel('Time')
        self.windowNeck.set_ylabel('Distant')

        # plt.show()
        self.times = []
        self.recordNeck = []
        self.recordHIP = []
        self.recordNeck_Rshoulder = []
        self.recordTimeList = []
        self.globalTime = 0
        self.fps_time = 0
        self.highestNeck = 0
        self.recordTimeNeckHighest = 0
        self.scaleFalling = 3000
        self.highestHIP = 0
        self.saveTimes = -1

        self.recordTimeHIPHighest = 0
        self.surpriseMovingTime = -1
        self.detectedHIP_Y = 0
        self.detectedNECK_Y = 0
        self.extraDistance = 0

        model = 'mobilenet_thin_432x368'
        w, h = model_wh(model)
        #model = 'cmu'
        #w, h = 432, 368
        camera = 0  # 1 mean external camera , 0 mean internal camera
        self.e = TfPoseEstimator(get_graph_path(model), target_size=(w, h))
        self.cam = cv2.VideoCapture(camera)
        ret_val, image = self.cam.read(cv2.IMREAD_COLOR)
        try:
            self.mesh(image)
        except Exception as e:
            print(e)
    def reduceRecord(self) :
        self.recordNeck = self.recordNeck[200:]
        self.recordHIP = self.recordHIP[200:]
        self.times = self.times[200:]
        self.recordTimeList = self.recordTimeList[200:]
        self.recordNeck_Rshoulder = self.recordNeck_Rshoulder[200:]
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
    def addRecordNeck_RShoulder(self,length):
        self.recordNeck_Rshoulder = self.recordNeck_Rshoulder+[length]
    def lengthBetweenPoint(self,pointA,pointB):
        x = (pointA[0] - pointB[0])**2
        y = (pointA[1] - pointB[1])**2
        return (abs(x - y)**(1/2))
    def setScaleFalling(self):
        self.scaleFalling = self.highestHIP - self.highestNeck
    def indexLastNumberMinValueList(self,listA,number):
        last = number
        minValueIndex = -1
        minValue = -1
        for i in range(len(listA)-last,len(listA)):
            if minValueIndex == -1 or listA[i] < minValue:
                minValue = listA[i]
                minValueIndex = i
        return minValueIndex
    def destroyAllRecord(self):
        self.times = []
        self.recordNeck = []
        self.recordHIP = []
        self.recordTimeList = []
        self.recordNeck_Rshoulder = []
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
        print('scaleFalling GOAL: [neck - hip ] ',abs(self.scaleFalling))

        print('HIGHEST NECK',self.highestNeck)
        print('current NECK',self.getLastNeck())
        print('result [ neck ]current - HIGHEST: ',abs(self.getLastNeck() - self.highestNeck))
        print('-------------------------------!!!!falling!!!!!!-----------------')
        print('-------------------------------!!!!falling!!!!!!-----------------')
        self.surpriseMovingTime = self.globalTime
        self.saveTimes = self.times[-1]
        #low value then far from camera
        print('set extraDistance')
        print(min(self.recordNeck_Rshoulder[-7:-2]))
        print(self.recordNeck_Rshoulder[-1])
        minNeckRShoulder = min(self.recordNeck_Rshoulder[-7:-2])
        if self.recordNeck_Rshoulder[-1] > minNeckRShoulder:
            print('ENTER CAMERA')
            self.extraDistance = (self.detectedHIP_Y - self.detectedNECK_Y)

        else:
            print('OUT CAMERA')
            rate = self.recordNeck_Rshoulder[-1]/minNeckRShoulder
            self.extraDistance = rate*(self.detectedHIP_Y - self.detectedNECK_Y)
            print(rate)

        print('extraDis : ',self.extraDistance)
        self.extraDistance = (self.detectedHIP_Y - self.detectedNECK_Y)*(1/4)
        print('set complete ')
    def countdownFalling(self):
        print('----------------------------------------')
        print('StartTime From: ',self.surpriseMovingTime)
        print('!!!!!Countdown[10] : ',self.globalTime - self.surpriseMovingTime,'!!!!!')
        print('would like to Cancel Countdown \nTake your neck to same level as NECK , HIP : ',self.detectedNECK_Y,self.detectedHIP_Y)
        print('current your NECK : ',self.getLastNeck())
        print('extraTotal:',self.detectedHIP_Y+self.extraDistance)
        print('----------------------------------------')
        #maybe not Falling but make sure with NECK last must move up to this position
        print('check STATE 2')
    def resetSurpriseMovingTime(self):
        self.surpriseMovingTime=-1
    def foundFalling(self):
        print('----------------------------------------')
        print('+++++FALL_DETECTED+++++++')
        print('----------------------------------------')
        time.sleep(10)
    def getLastNeck(self):
        return self.recordNeck[-1]
    def getLastTimes(self):
        return self.times[-1]
    def mesh(self, image):

        image_h, image_w = image.shape[:2]
        width = 300
        height = 300
        self.resetBitFalling()
        print('start-inderence',time.time())
        humans = self.e.inference(image, scales=[None])
        print('end-inderence',time.time())
        package = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
        self.globalTime = time.time()  #time of after drawing
        #camera not found NECK more than 10 second then reset list
        if self.globalTime - self.getLastRecordTime() >= 12:
            print('RESET STABLE,RECORDNECK,HIP,etc. [complete 12 second]')
            self.destroyAllRecord()

        image = package[0]
        status_part_body_appear = package[1]
        center_each_body_part = package[2]
        print('end Initialize mesh')
        #find length of neck , R_SHOULDER
        if 2 in center_each_body_part and 1 in center_each_body_part:
            p1 = center_each_body_part[2]
            p2 = center_each_body_part[1]
            self.addRecordNeck_RShoulder(self.lengthBetweenPoint(p1,p2))
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
        print('insert FPS')
        timeSave = time.time()
        if timeSave - self.fps_time > 0:
            self.addFPStoWindow(image,timeSave)
        print('show image')
        image = cv2.resize(image, (width, height))
        cv2.imshow('tf-pose-estimation result', image)
        self.fps_time = time.time()
        #UPDATE highest y point NECK  every 1
        #print('TIME : ',time.time() - self.recordTimeNeckHighest)
        print('start record everything')
        if 1 in center_each_body_part and (11 in center_each_body_part or 8 in center_each_body_part):
            self.setScaleFalling()
        if 1 in center_each_body_part and self.globalTime - self.getLastRecordTime() >= 0.25:  # every 0.3 second record
            print(self.globalTime - self.getLastRecordTime())
            self.addCountTimes()
            self.addRecordTime(self.globalTime)
            if len(self.recordNeck) > 600:
                self.reduceRecord()
            self.addRecordNeck(center_each_body_part[1][1])
            if 11 in center_each_body_part:
                self.addRecordHIP(center_each_body_part[11][1])
            elif 8 in center_each_body_part:
                self.addRecordHIP(center_each_body_part[8][1])

        print('find highest neck , hip')
        if len(self.recordNeck)>1:
            if (self.getLastNeck() < self.highestNeck) or (self.globalTime - self.recordTimeNeckHighest >= 0.25):
                self.recordTimeNeckHighest = self.globalTime
                #found last 6  min value

                index = self.indexLastNumberMinValueList(self.recordNeck,6)
                print('find index')
                self.highestNeck = self.recordNeck[index] #more HIGH more low value
                if len(self.recordHIP)>1:
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
        print('processing falling ---------')
        # print('NECK : -',self.recordNeck)
        # print('HIP : -',self.recordHIP)
        if self.highestHIP!=0 and len(self.recordNeck)>1 and 1 in center_each_body_part and self.surpriseMovingTime==-1:
            #NECK new Y point > NECK lastest Y point      falling
            #high , y low     || low , y high
            print('scaleFalling GOAL: [neck - hip ] ',abs(self.scaleFalling),'HIP,NECK',self.highestHIP,self.highestNeck)
            print('result [ neck ]current - HIGHEST: ',abs(self.getLastNeck() - self.highestNeck))
            print('Top NECk ',self.highestNeck,'  Last Neck ',self.getLastNeck())
            if (self.getLastNeck() > self.highestNeck) and (self.getLastNeck() - self.highestNeck )> abs(self.scaleFalling):
                self.detecedFirstFalling()

        elif self.surpriseMovingTime!=-1:
            self.countdownFalling()
            print('times - times : ',self.times[-1] - self.saveTimes)
            if self.times[-1] - self.saveTimes >= 2 and (self.getLastNeck() <= self.detectedNECK_Y or self.getLastNeck() <= (self.detectedHIP_Y+self.extraDistance)):
                print('---------------------------------------')
                print('Recover From STATE')
                print('---------------------------------------')
                self.resetSurpriseMovingTime()
            elif self.globalTime - self.surpriseMovingTime >= 10:
                self.setFalling()
                self.resetSurpriseMovingTime()
        print('end processing falling end mash()')

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
            print('NEWROUND')
            self.mesh(image)
            print('--generateGraphStable--')
            self.generateGraphStable()
            print('COMPLETE-')
        except Exception as e:
            print('ERROR : -> ',e)
            #print('body not in image')
    def generateGraphStable(self):
        plt.cla()
        self.windowNeck.set_ylim([0, 1500])
        plt.yticks(range(0, 1501, 100), fontsize=14)
        plt.xlim(0,600)
        plt.plot(self.times, self.recordNeck)
        print('--- Times : ',self.getLastTimes(),'||| plot at Time : ',self.getLastRecordTime(),'||| Value : ',self.getLastNeck())
        plt.pause(0.01)

    def animation(self):
        while True:
            self.update()
            if cv2.waitKey(1) == ord('q'):
                self.cam.release()
                cv2.destroyAllWindows()
                break

if __name__ == '__main__':
    # os.chdir('..')
    style.use('ggplot')
    t = Terrain()
    t.animation()
