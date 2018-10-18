
import sys
import cv2
import time
#import os

from estimator import TfPoseEstimator
from networks import get_graph_path, model_wh
from matplotlib import style
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import common

class Terrain(object):

    def __init__(self):
        """
        Initialize the graphics window and mesh surface
        """
        self.bitFalling = 0
        # Initialize plot.
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
        self.saveTimesStartFalling = -1

        self.recordTimeHIPHighest = 0
        self.surpriseMovingTime = -1
        self.detectedHIP_Y = 0
        self.detectedNECK_Y = 0
        self.extraDistance = 0

        self.fgbg = cv2.createBackgroundSubtractorMOG2(history=0,varThreshold=16,detectShadows=False)
        self.secondNeck = 0
        self.human_in_frame = False
        self.lastTimesFoundNeck = -1
        self.width = 300
        self.height = 300
        self.quotaVirtureNeck = 2
        self.used_quotaVirtureNeck = 0
        model = 'mobilenet_thin_432x368'
        w, h = model_wh(model)
        #model = 'cmu'
        #w, h = 432, 368
        camera = 0  # 1 mean external camera , 0 mean internal camera
        self.e = TfPoseEstimator(get_graph_path(model), target_size=(w, h))
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
    def destroyAll(self):
        self.times = []
        self.recordNeck = []
        self.recordHIP = []
        self.recordTimeList = []
        self.recordNeck_Rshoulder = []
        self.resetSurpriseMovingTime()
        self.resetBitFalling()
    # def addFPStoWindow(self,window,timeSave):
    #     cv2.putText(window,
    #                 "FPS: %f [press 'q'to quit]" % (
    #                     1.0 / (timeSave - self.fps_time)),
    #                 (10, 20),  cv2.FONT_HERSHEY_SIMPLEX, 1,
    #                 (0, 255, 0), 2)
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
        self.saveTimesStartFalling = self.times[-1]
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
        self.extraDistance = (self.detectedHIP_Y - self.detectedNECK_Y)*(2/4)
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
    def getLastNeck(self):
        return self.recordNeck[-1]
    def getLastTimes(self):
        return self.times[-1]
    def getSecondNeck(self):
        return self.secondNeck
    def getLastTimesFoundNeck(self):
        return self.lastTimesFoundNeck
    def savesecondNeck(self,image):
        blur = cv2.GaussianBlur(image, (5, 5), 0)
        fgmask = self.fgbg.apply(blur)
        cnts = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL,
    		cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]

    	# loop over the contours
        x_left = -1
        y_left = -1
        x_right = -1
        y_right = -1
        for c in cnts:
    		# if the contour is too small, ignore it
            # if cv2.contourArea(c) > 500:
            #     continue

    		# compute the bounding box for the contour, draw it on the frame,
    		# and update the text

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
            # cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
        if (x_left==0 and y_left==0 and x_right==self.width and y_right==self.height)==False:
            # cv2.rectangle(image, (x_left, y_left), (x_right, y_right), (0, 255, 0), 2)
            if self.human_in_frame:
                self.secondNeck = y_left+10
        # cv2.imshow('na',fgmask)
    def mesh(self, image):
        image = common.read_imgfile(image,None,None)
        self.resetBitFalling()
        self.savesecondNeck(image)
        print('start-inderence',time.time())
        humans = self.e.inference(image, scales=[None])
        print('end-inderence',time.time())
        package = TfPoseEstimator.draw_humans_adpt(image, humans, imgcopy=False)
        self.globalTime = time.time()  #time of after drawing
        image = package[0]
        #status_part_body_appear = package[1]
        center_each_body_part = package[2]
        #camera not found NECK more than 10 second then reset list
        if self.globalTime - self.getLastRecordTime() >= 12:
            print('RESET STABLE,RECORDNECK,HIP,etc. [complete 12 second]')
            self.destroyAll()
        if self.globalTime - self.getLastRecordTime() >= 2:
            print('maybe NECK or HUMAN not found [complete 1.5 second]')
            self.human_in_frame=False
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

        #UPDATE highest y point NECK  every 1
        #print('TIME : ',time.time() - self.recordTimeNeckHighest)
        print('start record everything')
        if 1 in center_each_body_part and (11 in center_each_body_part or 8 in center_each_body_part):
            self.setScaleFalling()
        if self.globalTime - self.getLastRecordTime() >= 0.25 :  # every 0.3 second record
            if 1 in center_each_body_part:
                print(self.globalTime - self.getLastRecordTime())
                self.addCountTimes()
                self.addRecordTime(self.globalTime)
                self.human_in_frame = True
                self.lastTimesFoundNeck =self.getLastTimes()
                self.used_quotaVirtureNeck=0
                self.addRecordNeck(center_each_body_part[1][1])
                if 11 in center_each_body_part:
                    self.addRecordHIP(center_each_body_part[11][1])
                elif 8 in center_each_body_part:
                    self.addRecordHIP(center_each_body_part[8][1])
            elif self.getLastTimesFoundNeck()==self.getLastTimes() and self.used_quotaVirtureNeck<=self.quotaVirtureNeck:
                print(self.globalTime - self.getLastRecordTime())
                self.addCountTimes()
                self.addRecordTime(self.globalTime)
                self.lastTimesFoundNeck =self.getLastTimes()
                self.addRecordNeck(self.getSecondNeck())
                print('addSecond Neck')
                self.used_quotaVirtureNeck+=1
            if len(self.recordNeck) > 600:
                self.reduceRecord()
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
        if self.highestHIP!=0 and len(self.recordNeck)>1 and self.surpriseMovingTime==-1:
            #NECK new Y point > NECK lastest Y point      falling
            #high , y low     || low , y high
            print('scaleFalling GOAL: [neck - hip ] ',abs(self.scaleFalling),'HIP,NECK',self.highestHIP,self.highestNeck)
            print('result [ neck ]current - HIGHEST: ',abs(self.getLastNeck() - self.highestNeck))
            print('Top NECk ',self.highestNeck,'  Last Neck ',self.getLastNeck())
            if (self.getLastNeck() > self.highestNeck) and (self.getLastNeck() - self.highestNeck )> abs(self.scaleFalling):
                self.detecedFirstFalling()

        elif self.surpriseMovingTime!=-1:
            self.countdownFalling()
            print('times - times : ',self.times[-1] - self.saveTimesStartFalling)
            if self.times[-1] - self.saveTimesStartFalling >= 2 and (self.getLastNeck() <= self.detectedNECK_Y or self.getLastNeck() <= (self.detectedHIP_Y+self.extraDistance)):
                print('---------------------------------------')
                print('Recover From STATE')
                print('---------------------------------------')
                self.resetSurpriseMovingTime()
            elif self.globalTime - self.surpriseMovingTime >= 10:
                self.setFalling()
                self.resetSurpriseMovingTime()
                self.destroyAll()
        print('end processing falling end mash()')

    def setFalling(self):
        self.bitFalling = 1
    def getBitFalling(self):
        return self.bitFalling
    def resetBitFalling(self):
        self.bitFalling = 0
if __name__ == '__main__':
    # os.chdir('..')
    style.use('ggplot')
    t = Terrain()
