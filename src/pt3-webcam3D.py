"""
This serve as our base openGL class.
"""

import numpy as np
import pyqtgraph.opengl as gl
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui
import sys

import cv2
import time
#import os

from estimator import TfPoseEstimator
from networks import get_graph_path, model_wh
from matplotlib import style
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import common
import mpl_toolkits.mplot3d.axes3d as p3
from lifting.prob_model import Prob3dPose
from lifting.draw import plot_pose
from lifting.draw import plot_pose_adapt



class Terrain(object):
    
    def __init__(self):
        """
        Initialize the graphics window and mesh surface
        """
        # Initialize plot.
        plt.ion()
        f = plt.figure(figsize=(5,5))
        f2 =plt.figure(figsize=(6,5))
        
        self.window3DBody = f.gca(projection='3d')
        self.window3DBody.set_title('3D_Body')
        self.windowStable = f2.add_subplot(1,1,1)
        self.windowStable.set_title('Stable')
        self.windowStable.set_xlabel('Time')
        self.windowStable.set_ylabel('Distant')
        self.windowStable.set_ylim([0,1500])
        
        #plt.show()
        self.times = [0]
        self.stable = [0]
        self.recordHead = []
        self.fps_time = 0

        model = 'mobilenet_thin_432x368'
        w, h = model_wh(model)
        #model = 'cmu'
        #w,h = 656,368
        camera = 1  #1 mean external camera , 0 mean internal camera

        self.lines = {}
        self.connection = [
        [0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8],
        [8, 9], [9, 10], [8, 11], [11, 12], [12, 13], [8, 14], [14, 15],
        [15, 16]
        ]
        self.e = TfPoseEstimator(get_graph_path(model), target_size=(w, h))
        self.cam = cv2.VideoCapture(camera)
        ret_val, image = self.cam.read(cv2.IMREAD_COLOR)
        
        self.poseLifting = Prob3dPose('./src/lifting/models/prob_model_params.mat')
        self.statusBodyWindow=0
        try:
            keypoints = self.mesh(image)
            
        except:
            pass
            
        
    
    def mesh(self, image):
        image_h, image_w = image.shape[:2]
        width = 300
        height = 300
        pose_2d_mpiis = []
        visibilities = []
        zoom = 1.0
        if zoom < 1.0:
            canvas = np.zeros_like(image)
            img_scaled = cv2.resize(image, None, fx=args.zoom, fy=args.zoom, interpolation=cv2.INTER_LINEAR)
            dx = (canvas.shape[1] - img_scaled.shape[1]) // 2
            dy = (canvas.shape[0] - img_scaled.shape[0]) // 2
            canvas[dy:dy + img_scaled.shape[0], dx:dx + img_scaled.shape[1]] = img_scaled
            image = canvas
        elif zoom > 1.0:
            img_scaled = cv2.resize(image, None, fx=args.zoom, fy=args.zoom, interpolation=cv2.INTER_LINEAR)
            dx = (img_scaled.shape[1] - image.shape[1]) // 2
            dy = (img_scaled.shape[0] - image.shape[0]) // 2
            image = img_scaled[dy:image.shape[0], dx:image.shape[1]]

    
        
        humans = self.e.inference(image, scales=[None])
        package = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
        
        
        image = package[0]
        status_part_body_appear = package[1]
        name_part_body = ["Nose",
                          "Neck",
                          "RShoulder",
                          "RElbow",
                          "RWrist",
                          "LShoulder",
                          "LElbow",
                          "LWrist",
                          "RHip",
                          "RKnee",
                          "RAnkle",
                          "LHip",
                          "LKnee",
                          "LAnkle",
                          "REye",
                          "LEye",
                          "REar",
                          "LEar",
                          ]
        detected_part = []
        for human in humans:
            pose_2d_mpii, visibility = common.MPIIPart.from_coco(human)
            pose_2d_mpiis.append(
                [(int(x * width + 0.5), int(y * height + 0.5)) for x, y in pose_2d_mpii]
            )
            visibilities.append(visibility)
        
        cv2.putText(image,
                    "FPS: %f [press 'q'to quit]" % (1.0 / (time.time() - self.fps_time)),
                    (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)
        image = cv2.resize(image,(width,height))
        cv2.imshow('tf-pose-estimation result',image)
        
        pose_2d_mpiis = np.array(pose_2d_mpiis)
        visibilities = np.array(visibilities)
        transformed_pose2d, weights = self.poseLifting.transform_joints(pose_2d_mpiis, visibilities)
        pose_3d = self.poseLifting.compute_3d(transformed_pose2d, weights)
        for i, single_3d in enumerate(pose_3d):
            #plot_pose(single_3d)
            plot_pose_adapt(single_3d,self.window3DBody)
        self.fps_time = time.time()
        
        
        #Matt plot lib
        #print(pose_3d)
        #----------------------------------------
        #pyQT graph
        pose_3dqt = np.array(pose_3d[0]).transpose()
        
        bodyPartName = ['C_Hip',
                        'R_Hip',
                    'R_Knee',
                        'R_Ankle',
                        'L_Hip',
                    'L_Knee',
                        'L_Ankle',
                        'Center',
                    'C_Shoulder',
                        'Neck',
                        'Head',
                    'L_Shoulder',
                        'L_Elbow',
                        'L_Wrist',
                    'R_Shoulder',
                        'R_Elbow',
                        'R_Wrist']
        #for part in range(len(pose_3dqt)):
        #    print(bodyPartName[part],pose_3dqt[part])
            
        #for id_part in range(len(status_part_body_appear)):
            #check this part body appear or not
         #   if status_part_body_appear[id_part] == 1:
          #      print("%-10s"%name_part_body[id_part],": appear")
           #     detected_part.append(id_part)
            #else:
             #   print("%-10s"%name_part_body[id_part],": disappear")
        #list_to_check_fall_deteced = [[1,8]  , #neck,RHIP
        #                                [1,9],  #neck RKnee
        #                                [1,10], #neck RAnkle
        #                                [1,11], #neck LHip
        #                                [1,12], #neck LKne e
        #                                [1,13]]  #neck LAnkle
        if int(self.fps_time)%1==0: #every 1 second record
            self.times = self.times + [self.times[-1]+1]
            if len(self.stable)>1000:
                self.stable = self.stable[200:]
                self.recordHead = self.recordHead[200:]
            if self.stable == [0]:
                self.stable = self.stable + [0]
                self.recordHead = [pose_3dqt[10][2]] + [pose_3dqt[10][2]]
            else:
                #highest 800  , 550-600 average
                self.stable = self.stable + [abs(pose_3dqt[10][2] - self.recordHead[-1])]
                self.recordHead = self.recordHead + [pose_3dqt[10][2]]
            
            
        status_found = 0
        for id_part in detected_part:
            #if id_part in [8,9,10,11,12,13] and 1 in detected_part:
            #    status_found = 1
            if id_part in [8,11] and 1 in detected_part:
                status_found = 1
        if status_found :
            print("-------Ready for detece--------")
            if self.fall_detection(pose_3dqt):
                print("-----\nFOUND !!!\n-----")
        #----------
        keypoints = pose_3d[0].transpose()

        return keypoints / 80
    def fall_detection(self,pose_3dqt):
        print("VALUE Z : ","NECK , C_HIP",((pose_3dqt[10][2] - pose_3dqt[0][2])**2)**(1/2))
        #print("VALUE Z : ","NECK , R_HIP",((pose_3dqt[10][2] - pose_3dqt[1][2])**2)**(1/2))
        #print("VALUE Z : ","NECK , R_KNEE",((pose_3dqt[10][2] - pose_3dqt[2][2])**2)**(1/2))
        #print("VALUE Z : ","NECK , R_ANKLE",((pose_3dqt[10][2] - pose_3dqt[3][2])**2)**(1/2))
        #print("VALUE Z : ","NECK , L_HIP",((pose_3dqt[10][2] - pose_3dqt[4][2])**2)**(1/2))
        #print("VALUE Z : ","NECK , L_KNEE",((pose_3dqt[10][2] - pose_3dqt[5][2])**2)**(1/2))
        #print("VALUE Z : ","NECK , L_ANKLE",((pose_3dqt[10][2] - pose_3dqt[6][2])**2)**(1/2))
        #NECK C_HIP
        if ((pose_3dqt[10][2] - pose_3dqt[0][2])**2)**(1/2)  <= 200:
            return True
        #NECK R_HIP    Z-graph
        #elif ((pose_3dqt[10][2] - pose_3dqt[1][2])**2)**(1/2)  <= 200:
        #    return True
        #NECK  R_Knee
        #elif ((pose_3dqt[10][2] - pose_3dqt[2][2])**2)**(1/2)  <= 200:
        #    return True
        #NECK RAnkle
        #elif ((pose_3dqt[10][2] - pose_3dqt[3][2])**2)**(1/2)  <= 200:
        #    return True
        #NECK LHip
        #elif ((pose_3dqt[10][2] - pose_3dqt[4][2])**2)**(1/2)  <= 200:
        #    return True
        #NECK LKnee
        #elif ((pose_3dqt[10][2] - pose_3dqt[5][2])**2)**(1/2)  <= 200:
        #    return True
        #NECK LAnkle
        #elif ((pose_3dqt[10][2] - pose_3dqt[6][2])**2)**(1/2)  <= 200:
        #    return True
        return False
    def update(self):
        """
        update the mesh and shift the noise each time
        """
        ret_val, image = self.cam.read()
        try:
            keypoints = self.mesh(image)
            self.generateGraphStable()
            
        except AssertionError:
            print('body not in image')
        else:
            pass
            
    def generateGraphStable(self):            
        plt.plot(self.times, self.stable)
        plt.pause(0.1)
        
    def start(self):
        """
        get the graphics window open and setup
        """
        #if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            #QtGui.QApplication.instance().exec_()

    def animation(self):
        while True:
            self.update()
            if cv2.waitKey(1)==ord('q'):
                self.cam.release()
                cv2.destroyAllWindows()
                break
    


if __name__ == '__main__':
    #os.chdir('..')
    style.use('ggplot')
    t = Terrain()
    t.animation()
