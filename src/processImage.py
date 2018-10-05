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
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import common
import mpl_toolkits.mplot3d.axes3d as p3
from lifting.prob_model import Prob3dPose
from lifting.draw import plot_pose
from lifting.draw import plot_pose_adapt



class Terrain(object):
    
    def __init__(self):
        
        self.times = [0]
        self.stable = [0]
        self.recordNeck = []
        self.recordTime = 0
        self.fps_time = 0

        model = 'mobilenet_thin_432x368'
        w, h = model_wh(model)
        #model = 'cmu'
        #w, h = 656, 368

        self.lines = {}
        self.connection = [
        [0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8],
        [8, 9], [9, 10], [8, 11], [11, 12], [12, 13], [8, 14], [14, 15],
        [15, 16]
        ]
        self.e = TfPoseEstimator(get_graph_path(model), target_size=(w, h))
        #self.image = common.read_imgfile('./images/tet.jpg',None,None)
        
        self.poseLifting = Prob3dPose('./src/lifting/models/prob_model_params.mat')
        self.statusBodyWindow=0
        #try:
            #keypoints = self.mesh(self.image)
            #plt.imshow(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
            #plt.show()
        #except:
            #pass
    def show3DImage(self):
        for i, single_3d in enumerate(self.pose_3d):
            plot_pose(single_3d)
        plt.imshow(self.image)
        plt.show()
    def mesh(self, image):
        self.image = common.read_imgfile(image,None,None)
        image_h, image_w = self.image.shape[:2]
        width = 300
        height = 300
        pose_2d_mpiis = []
        visibilities = []
        zoom = 1.0
        if zoom < 1.0:
            canvas = np.zeros_like(self.image)
            img_scaled = cv2.resize(self.image, None, fx=args.zoom, fy=args.zoom, interpolation=cv2.INTER_LINEAR)
            dx = (canvas.shape[1] - img_scaled.shape[1]) // 2
            dy = (canvas.shape[0] - img_scaled.shape[0]) // 2
            canvas[dy:dy + img_scaled.shape[0], dx:dx + img_scaled.shape[1]] = img_scaled
            self.image = canvas
        elif zoom > 1.0:
            img_scaled = cv2.resize(self.image, None, fx=args.zoom, fy=args.zoom, interpolation=cv2.INTER_LINEAR)
            dx = (img_scaled.shape[1] - self.image.shape[1]) // 2
            dy = (img_scaled.shape[0] - self.image.shape[0]) // 2
            self.image = img_scaled[dy:self.image.shape[0], dx:self.image.shape[1]]

    
        
        humans = self.e.inference(self.image, scales=[None])
        package = TfPoseEstimator.draw_humans(self.image, humans, imgcopy=False)
        
        
        self.image = package[0]
        
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
        
        pose_2d_mpiis = np.array(pose_2d_mpiis)
        visibilities = np.array(visibilities)
        transformed_pose2d, weights = self.poseLifting.transform_joints(pose_2d_mpiis, visibilities)
        self.pose_3d = self.poseLifting.compute_3d(transformed_pose2d, weights)
        
        pose_3dqt = np.array(self.pose_3d[0]).transpose()
        
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
        
        if time.time() - self.recordTime >= 0.3: #every 1 second record
            self.times = self.times + [self.times[-1]+1]
            self.recordTime = time.time()
            if len(self.stable)>1000:
                self.stable = self.stable[200:]
                self.recordNeck = self.recordNeck[200:]
            if self.stable == [0]:
                self.stable = self.stable + [0]
                self.recordNeck = [pose_3dqt[9][2]] + [pose_3dqt[9][2]]
            else:
                #550-600 average
                self.stable = self.stable + [abs(pose_3dqt[9][2] - self.recordHead[-1])]
                self.recordNeck = self.recordNeck + [pose_3dqt[9][2]]
            
        
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
        keypoints = self.pose_3d[0].transpose()

        return keypoints / 80
    def get_graph_stable(self):
        return self.stable
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
    


if __name__ == '__main__':
    os.chdir('..')
    style.use('ggplot')
    #t = Terrain()
