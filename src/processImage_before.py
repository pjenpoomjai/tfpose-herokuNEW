
import sys
import cv2
import time
#import os

from estimator import TfPoseEstimator
from networks import get_graph_path, model_wh
from matplotlib import style
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import imutils
import common
class Terrain(object):

    def __init__(self):
        """
        Initialize the graphics window and mesh surface
        """
        self.bitFalling = 0
        # Initialize plot.
        self.times = []
        self.recordVelocity = [0]
        self.recordNeck = []
        self.recordYTopRectangle = []
        self.recordHIP = []
        self.recordTimeList = []
        self.globalTime = 0
        self.highestNeck = 0
        self.hightestNeckTime = 0
        self.highestHIP = 0
        self.saveTimesStartFalling = -1

        self.surpriseMovingTime = -1
        self.detectedHIP_Y = 0
        self.detectedNECK_Y = 0
        self.extraDistance = 0

        self.fgbg = cv2.createBackgroundSubtractorMOG2(history=1,varThreshold=400,detectShadows=False)
        self.secondNeck = 0
        self.human_in_frame = False
        self.lastTimesFoundNeck = -1
        self.width = 300
        self.height = 300
        self.quotaVirtureNeck = 3
        self.used_quotaVirtureNeck = 0
        model = 'mobilenet_thin_432x368'
        w, h = model_wh(model)
        self.e = TfPoseEstimator(get_graph_path(model), target_size=(w, h))
    def reduceRecord(self) :
        self.recordNeck = self.recordNeck[-100:]
        self.recordHIP = self.recordHIP[-100:]
        self.times = self.times[-100:]
        self.recordVelocity = self.recordVelocity[-100:]
        self.recordTimeList = self.recordTimeList[-100:]
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
    def destroyAll(self):
        self.times = []
        self.recordNeck = []
        self.recordHIP = []
        self.recordTimeList = []
        self.recordVelocity = [0]
        self.recordYTopRectangle = []
        self.resetSurpriseMovingTime()
        self.resetBitFalling()
    def detecedFirstFalling(self):
        self.detectedNECK_Y = self.highestNeck
        self.detectedHIP_Y  = self.highestHIP
        print('-----!!!!falling!!!!!!-------')
        self.surpriseMovingTime = self.globalTime
        self.saveTimesStartFalling = self.times[-1]
        print('set extraDistance')
        self.extraDistance = (self.detectedHIP_Y - self.detectedNECK_Y)*(1/2)
        print('extraDis : ',self.extraDistance)
        print('set complete ')
    def countdownFalling(self):
        # print('StartTime From: ',self.surpriseMovingTime)
        print('!!!!!Countdown[10] : ',self.globalTime - self.surpriseMovingTime,'!!!!!')
        # print('would like to Cancel Countdown \nTake your neck to same level as NECK , HIP : ',self.detectedNECK_Y,self.detectedHIP_Y)
        # print('current your NECK : ',self.getLastNeck())
        # print('extraTotal:',self.detectedHIP_Y+self.extraDistance)
        #maybe not Falling but make sure with NECK last must move up to this position
        # print('Check in second stage.')
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
        cnts = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
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
        if (x_left==0 and y_left==0 and x_right==self.width and y_right==self.height)==False:
            if self.human_in_frame and y_left != -1:
                # cv2.rectangle(image, (x_left, y_left), (x_right, y_right), (0, 255, 0), 2)
                self.secondNeck = y_left
                print('second Neck : ',self.secondNeck)
                self.recordYTopRectangle = self.recordYTopRectangle + [self.secondNeck]
        # cv2.imshow('na',fgmask)
    def processFall(self,image):
        print('processing falling ---------')
        print('highestNeck',self.highestNeck)
        print('highestHIP',self.highestHIP)
        print('time duration : ',(self.recordTimeList[-1] - self.recordTimeList[-2]))
        if self.highestHIP!=0 and len(self.recordNeck)>1 and self.surpriseMovingTime==-1 :
            #NECK new Y point > NECK lastest Y point      falling
            #high , y low     || low , y high
            print('LAST_NECK',self.getLastNeck(),'HIGHTEST_HIP', self.highestHIP)
            print('max-Velocity :',max(self.recordVelocity[-6:]))
            vHumanFall = max(self.recordVelocity[-6:])
            # vHumanFall = abs((self.highestHIP - self.highestNeck) / (self.recordTimeList[-1] - self.highestNeckTime))
            timeFall = 0.5
            vThresholdA = int(abs((self.highestHIP - self.highestNeck)) / (timeFall))
            # vThresholdAB = int(abs((self.highestHIP - self.highestNeck)) / (0.45))
            # vThresholdB = int(abs((self.highestHIP - self.highestNeck)) / (0.4))
            # vThresholdC = int(abs((self.highestHIP - self.highestNeck)) / (0.3))
            print('vHumanFall',vHumanFall,' >= vThA :', vThresholdA)
            # print('vHumanFall',vHumanFall,' >= vThA+B :', vThresholdAB)
            # print('vHumanFall',vHumanFall,' >= vThB :', vThresholdB)
            # print('vHumanFall',vHumanFall,' >= vThC :', vThresholdC)
            if self.getLastNeck() >= self.highestHIP :
                print('~~falling~~')
                if vHumanFall >= vThresholdA:
                    self.detecedFirstFalling()
                # if vHumanFall >= vThresholdAB:
                #     self.detecedFirstFalling()
                #     cv2.line(image, (0, self.getLastNeck()), (self.height,self.getLastNeck()), (0, 255, 0), 2)
                #     cv2.imshow('shotFall_lastNECK_0.45',image)
                # if vHumanFall >= vThresholdB:
                #     cv2.line(image, (0, self.getLastNeck()), (self.height,self.getLastNeck()), (0, 255, 0), 2)
                #     cv2.imshow('shotFall_lastNECK_0.4',image)
                # if vHumanFall >= vThresholdC:
                #     cv2.line(image, (0, self.getLastNeck()), (self.height,self.getLastNeck()), (0, 255, 0), 2)
                #     cv2.imshow('shotFall_lastNECK_0.3',image)


        elif self.surpriseMovingTime!=-1:
            self.countdownFalling()
            if self.globalTime - self.surpriseMovingTime >= 2 and (self.getLastNeck() <= (self.detectedHIP_Y - self.extraDistance)):
                print('Recover From STATE')
                print('---------------------------------------')
                self.destroyAll()
            elif self.globalTime - self.surpriseMovingTime >= 10:
                print('Warning : Falling happening')
                self.setFalling()
                self.destroyAll()
    def mesh(self, image):
        image = common.read_imgfile(nameImage,None,None)
        print('start-inderence',time.time())
        humans = self.e.inference(image, scales=[None])
        print('end-inderence',time.time())
        self.resetBitFalling()
        self.savesecondNeck(image)
        package = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
        self.globalTime = time.time()  #time of after drawing
        image = package[0]
        #status_part_body_appear = package[1]
        center_each_body_part = package[2]
        #camera not found NECK more than 10 second then reset list
        if self.globalTime - self.getLastRecordTime() >= 12:
            print('RESET STABLE,RECORDNECK,HIP,etc. [complete 12 second]')
            self.destroyAll()
        if self.globalTime - self.getLastRecordTime() >= 2:
            print('maybe NECK or HUMAN not found [complete 2 second]')
            self.human_in_frame=False
        print('end Initialize mesh')
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
        print('start record everything')
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
        elif self.getLastTimesFoundNeck()==self.getLastTimes() and self.used_quotaVirtureNeck<self.quotaVirtureNeck and self.secondNeck >= self.getLastNeck():
            # print(self.globalTime - self.getLastRecordTime())
            self.addCountTimes()
            self.addRecordTime(self.globalTime)
            self.lastTimesFoundNeck =self.getLastTimes()
            self.addRecordNeck(self.getSecondNeck())
            self.addRecordVelocity(self.recordYTopRectangle,self.recordTimeList)
            print('addSecond Neck',self.used_quotaVirtureNeck)
            self.used_quotaVirtureNeck+=1
        if len(self.recordNeck) > 300:
            self.reduceRecord()
        # print('find highest neck , hip')
        minNumber = -1
        loop = 6
        if len(self.recordNeck) < loop:
            loop = len(self.recordNeck)
        for i in range(1,loop+1):
            if minNumber==-1 or self.recordNeck[-i] <= minNumber:
                self.highestNeck = self.recordNeck[-i] #more HIGH more low value
                self.hightestNeckTime = self.recordTimeList[-i]
                minNumber = self.recordNeck[-i]
        print('hightestTIMENECK', self.hightestNeckTime)
        if len(self.recordHIP)>1:
            #11 L_HIP
            if 11 in center_each_body_part:
                self.highestHIP = min(self.recordHIP[-6:])
            #8 R_HIP
            elif 8 in center_each_body_part:
                self.highestHIP = min(self.recordHIP[-6:])
        if len(self.recordNeck)>1:
            self.processFall(image)
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
