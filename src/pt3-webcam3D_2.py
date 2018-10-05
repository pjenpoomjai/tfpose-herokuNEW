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
        """
        Initialize the graphics window and mesh surface
        """
        # Initialize plot.
        plt.ion()
        f = plt.figure(figsize=(5, 5))
        f2 = plt.figure(figsize=(6, 5))

        self.window3DBody = f.gca(projection='3d')
        self.window3DBody.set_title('3D_Body')

        self.windowStable = f2.add_subplot(1, 1, 1)
        self.windowStable.set_title('Stable')
        self.windowStable.set_xlabel('Time')
        self.windowStable.set_ylabel('Distant')
        self.windowStable.set_ylim([0, 1500])
        plt.yticks(range(0, 1501, 100), fontsize=14)

        # plt.show()
        self.times = [0]
        self.stable = [0]
        self.recordNeck = []
        self.recordTime = 0
        self.fps_time = 0

        self.threshold = 100
        self.count_fall_detected = 0
        self.started_fall = False
        #model = 'mobilenet_thin_432x368'
        #w, h = model_wh(model)
        model = 'cmu'
        w, h = 656, 368
        camera = 0  # 1 mean external camera , 0 mean internal camera

        self.lines = {}
        self.connection = [
            [0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8],
            [8, 9], [9, 10], [8, 11], [11, 12], [12, 13], [8, 14], [14, 15],
            [15, 16]
        ]
        self.e = TfPoseEstimator(get_graph_path(model), target_size=(w, h))
        self.cam = cv2.VideoCapture(camera)
        ret_val, image = self.cam.read(cv2.IMREAD_COLOR)

        self.poseLifting = Prob3dPose(
            './src/lifting/models/prob_model_params.mat')
        self.statusBodyWindow = 0
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
            img_scaled = cv2.resize(
                image, None, fx=args.zoom, fy=args.zoom, interpolation=cv2.INTER_LINEAR)
            dx = (canvas.shape[1] - img_scaled.shape[1]) // 2
            dy = (canvas.shape[0] - img_scaled.shape[0]) // 2
            canvas[dy:dy + img_scaled.shape[0], dx:dx +
                   img_scaled.shape[1]] = img_scaled
            image = canvas
        elif zoom > 1.0:
            img_scaled = cv2.resize(
                image, None, fx=args.zoom, fy=args.zoom, interpolation=cv2.INTER_LINEAR)
            dx = (img_scaled.shape[1] - image.shape[1]) // 2
            dy = (img_scaled.shape[0] - image.shape[0]) // 2
            image = img_scaled[dy:image.shape[0], dx:image.shape[1]]

        humans = self.e.inference(image, scales=[None])
        package = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)

        image = package[0]
        status_part_body_appear = package[1]
        print(status_part_body_appear)
        name_part_body = ["Nose",  # 0
                          "Neck",  # 1
                          "RShoulder",  # 2
                          "RElbow",  # 3
                          "RWrist",  # 4
                          "LShoulder",  # 5
                          "LElbow",  # 6
                          "LWrist",  # 7
                          "RHip",  # 8
                          "RKnee",  # 9
                          "RAnkle",  # 10
                          "LHip",  # 11
                          "LKnee",  # 12
                          "LAnkle",  # 13
                          "REye",  # 14
                          "LEye",  # 15
                          "REar",  # 16
                          "LEar",  # 17
                          ]
        detected_part = []

        for human in humans:
            pose_2d_mpii, visibility = common.MPIIPart.from_coco(human)
            pose_2d_mpiis.append(
                [(int(x * width + 0.5), int(y * height + 0.5))
                 for x, y in pose_2d_mpii]
            )
            visibilities.append(visibility)

        cv2.putText(image,
                    "FPS: %f [press 'q'to quit]" % (
                        1.0 / (time.time() - self.fps_time)),
                    (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0), 2)
        image = cv2.resize(image, (width, height))
        cv2.imshow('tf-pose-estimation result', image)

        pose_2d_mpiis = np.array(pose_2d_mpiis)
        visibilities = np.array(visibilities)

        transformed_pose2d, weights = self.poseLifting.transform_joints(
            pose_2d_mpiis, visibilities)
        pose_3d = self.poseLifting.compute_3d(transformed_pose2d, weights)

        for i, single_3d in enumerate(pose_3d):
            # plot_pose(single_3d)
            plot_pose_adapt(single_3d, self.window3DBody)
        self.fps_time = time.time()

        # Matt plot lib
        # print(pose_3d)
        # ----------------------------------------
        # pyQT graph
        pose_3dqt = np.array(pose_3d[0]).transpose()
        print(pose_3dqt)
        bodyPartName = ['C_Hip',  # 0
                        'R_Hip',  # 1
                        'R_Knee',  # 2
                        'R_Ankle',  # 3
                        'L_Hip',  # 4
                        'L_Knee',  # 5
                        'L_Ankle',  # 6
                        'Center',  # 7
                        'C_Shoulder',  # 8
                        'Neck',  # 9
                        'Head',  # 10
                        'L_Shoulder',  # 11
                        'L_Elbow',  # 12
                        'L_Wrist',  # 13
                        'R_Shoulder',  # 14
                        'R_Elbow',  # 15
                        'R_Wrist']  # 16

        for id_part in range(len(status_part_body_appear)):
            # check this part body appear or not
            if status_part_body_appear[id_part] == 1:
                print("%-10s" % name_part_body[id_part], ": appear")
                detected_part.append(id_part)
            else:
                print("%-10s" % name_part_body[id_part], ": disappear")
        # list_to_check_fall_deteced = [[1,8]  , #neck,RHIP
        #                                [1,9],  #neck RKnee
        #                                [1,10], #neck RAnkle
        #                                [1,11], #neck LHip
        #                                [1,12], #neck LKne e
        #                                [1,13]]  #neck LAnkle

        if time.time() - self.recordTime >= 0.3:  # every 0.3 second record
            print(time.time() - self.recordTime)
            self.times = self.times + [self.times[-1]+1]
            self.recordTime = time.time()
            if len(self.stable) > 1000:
                self.stable = self.stable[200:]
                self.recordNeck = self.recordNeck[200:]
            if self.stable == [0]:
                self.stable = self.stable + [0]
                self.recordNeck = [pose_3dqt[9][2]] + [pose_3dqt[9][2]]
            else:
                # highest 800  , 550-600 average
                self.stable = self.stable + \
                    [abs(pose_3dqt[9][2] - self.recordNeck[-1])]
                self.recordNeck = self.recordNeck + [pose_3dqt[9][2]]

        isFalled = False

        if 1 in detected_part or 0 in detected_part:
            print("NECK or HEAD found")
            print("NEXK Z:     ", pose_3dqt[9][2])
            for id_part in detected_part:

                if id_part == 10:  # R_ANKLE APPEAR
                    print("-------Ready for detect--------")
                    # print("R_ankle ", abs(pose_3dqt[9][2] - pose_3dqt[3][2]))
                    if abs(pose_3dqt[9][2] - pose_3dqt[3][2]) <= self.threshold:
                        isFalled = True
                        self.started_fall = True
                elif id_part == 13:  # L_ANKLE APPEAR
                    print("-------Ready for detect--------")
                    if abs(pose_3dqt[9][2] - pose_3dqt[6][2]) <= self.threshold:
                        isFalled = True
                        self.started_fall = True
                elif id_part == 8:  # R_HIP APPEAR
                    print("-------Ready for detect--------")
                    # print("R_HIP ", abs(pose_3dqt[9][2] - pose_3dqt[1][2]))
                    if abs(pose_3dqt[9][2] - pose_3dqt[1][2]) <= self.threshold:
                        isFalled = True
                        self.started_fall = True
                elif id_part == 11:  # L_HIP APPEAR
                    print("-------Ready for detect--------")
                    print("L_HIP ", abs(pose_3dqt[9][2] - pose_3dqt[4][2]))
                    if abs(pose_3dqt[9][2] - pose_3dqt[4][2]) <= self.threshold:
                        isFalled = True
                        self.started_fall = True
                elif id_part == 9:  # R_Knee APPEAR
                    print("-------Ready for detect--------")
                    if abs(pose_3dqt[9][2] - pose_3dqt[2][2]) <= self.threshold:
                        isFalled = True
                        self.started_fall = True
                elif id_part == 12:  # L_Knee APPEAR
                    print("-------Ready for detect--------")
                    if abs(pose_3dqt[9][2] - pose_3dqt[5][2]) <= self.threshold:
                        isFalled = True
                        self.started_fall = True

        # R_Shoulder found
        elif 2 in detected_part:
            print("Shoulder found")
            for id_part in detected_part:

                if id_part == 10:  # R_ANKLE APPEAR
                    print("-------Ready for detect--------")
                    if abs(pose_3dqt[14][2] - pose_3dqt[3][2]) <= self.threshold:
                        isFalled = True
                        self.started_fall = True
                elif id_part == 13:  # L_ANKLE APPEAR
                    print("-------Ready for detect--------")
                    if abs(pose_3dqt[14][2] - pose_3dqt[6][2]) <= self.threshold:
                        isFalled = True
                        self.started_fall = True
                elif id_part == 8:  # R_HIP APPEAR
                    print("-------Ready for detect--------")
                    if abs(pose_3dqt[14][2] - pose_3dqt[1][2]) <= self.threshold:
                        isFalled = True
                        self.started_fall = True
                elif id_part == 11:  # L_HIP APPEAR
                    print("-------Ready for detect--------")
                    if abs(pose_3dqt[14][2] - pose_3dqt[4][2]) <= self.threshold:
                        isFalled = True
                        self.started_fall = True
                elif id_part == 9:  # R_Knee APPEAR
                    print("-------Ready for detect--------")
                    if abs(pose_3dqt[14][2] - pose_3dqt[2][2]) <= self.threshold:
                        isFalled = True
                        self.started_fall = True
                elif id_part == 12:  # L_Knee APPEAR
                    print("-------Ready for detect--------")
                    if abs(pose_3dqt[14][2] - pose_3dqt[5][2]) <= self.threshold:
                        isFalled = True
                        self.started_fall = True
        # L_Shoulder found
        elif 5 in detected_part:
            print("Shoulder found")
            for id_part in detected_part:

                if id_part == 10:  # R_ANKLE APPEAR
                    print("-------Ready for detect--------")
                    if abs(pose_3dqt[11][2] - pose_3dqt[3][2]) <= self.threshold:
                        isFalled = True
                        self.started_fall = True
                elif id_part == 13:  # L_ANKLE APPEAR
                    print("-------Ready for detect--------")
                    if abs(pose_3dqt[11][2] - pose_3dqt[6][2]) <= self.threshold:
                        isFalled = True
                        self.started_fall = True
                elif id_part == 8:  # R_HIP APPEAR
                    print("-------Ready for detect--------")
                    if abs(pose_3dqt[11][2] - pose_3dqt[1][2]) <= self.threshold:
                        isFalled = True
                        self.started_fall = True
                elif id_part == 11:  # L_HIP APPEAR
                    print("-------Ready for detect--------")
                    if abs(pose_3dqt[11][2] - pose_3dqt[4][2]) <= self.threshold:
                        isFalled = True
                        self.started_fall = True
                elif id_part == 9:  # R_Knee APPEAR
                    print("-------Ready for detect--------")
                    if abs(pose_3dqt[11][2] - pose_3dqt[2][2]) <= self.threshold:
                        isFalled = True
                        self.started_fall = True
                elif id_part == 12:  # L_Knee APPEAR
                    print("-------Ready for detect--------")
                    if abs(pose_3dqt[11][2] - pose_3dqt[5][2]) <= self.threshold:
                        isFalled = True
                        self.started_fall = True

        if self.started_fall:
            if isFalled:
                # send publish msg to Android
                self.count_fall_detected += 1
                print("------------- FALL counting --------------")

                if self.count_fall_detected == 1:
                    global start_time
                    start_time = time.time()

            elasped_time = time.time() - start_time
            if elasped_time >= 5:  # fall more than 5 second
                if self.count_fall_detected >= 2:
                    print(
                        "******************** FALL DETECTED !!!!! ******************************")

                self.started_fall = False
                self.count_fall_detected = 0

        keypoints = pose_3d[0].transpose()

        return keypoints / 80

    def fall_detection(self, pose_3dqt):

        # NECK C_HIP
        if abs(pose_3dqt[9][2] - pose_3dqt[0][2]) <= self.threshold:
            return True
        # NECK L_HIP
        elif abs(pose_3dqt[9][2] - pose_3dqt[4][2]) <= self.threshold:
            return True
        # NECK R_HIP
        elif abs(pose_3dqt[9][2] - pose_3dqt[1][2]) <= self.threshold:
            return True
        # NECK R_ANKLE
        elif abs(pose_3dqt[9][2] - pose_3dqt[3][2]) <= self.threshold:
            return True
         # NECK L_ANKLE
        elif abs(pose_3dqt[9][2] - pose_3dqt[6][2]) <= self.threshold:
            return True
         # NECK R_KNEE
        elif abs(pose_3dqt[9][2] - pose_3dqt[2][2]) <= self.threshold:
            return True
         # NECK L_KNEE
        elif abs(pose_3dqt[9][2] - pose_3dqt[5][2]) <= self.threshold:
            return True
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
        # if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        # QtGui.QApplication.instance().exec_()

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
