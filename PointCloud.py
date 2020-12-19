import math
import time

from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph.opengl as gl
import numpy as np
from pykinect2.PyKinectV2 import *
from pykinect2 import PyKinectV2
from pykinect2 import PyKinectRuntime
import mapper
import cv2
import sys
import os
import ctypes
from frame import Frame
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

class Cloud:

    def __init__(self):
        self._done = False

        self._bodies = None
        self.body_tracked = False
        self.joint_points = np.array([])
        self.joint_points3D = np.array([])
        self.joint_points_RGB = np.array([])
        self.joint_state = np.array([])
        self._frameRGB = None
        self._frameDepth = None
        self._frameDepthQuantized = None
        self._frameSkeleton = None
        self.frameNum = 0
        # Initialize Kinect object
        self._kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color|PyKinectV2.FrameSourceTypes_Depth|PyKinectV2.FrameSourceTypes_Body|PyKinectV2.FrameSourceTypes_BodyIndex)
        self._body_index = None
        self._body_index_points = None
        self._cloud = False
        self._depth = None
        self._color_frame = None
        self._red = 255  #
        self._green = 255
        self._blue = 255
        self._size = 0.1
        self._opacity = 0
        self._dt = .0  #
        self._skeleton_points = None
        self._color_point_cloud = False
        self._depth_point_cloud = True
        self._simultaneously_point_cloud = True
        self._skeleton_point_cloud = True
        self._dynamic = True
        self._body_index_cloud = False
        self._color_overlay = False
        self._dir_path = os.path.dirname(os.path.realpath(__file__))
        self._body_frame = None
        self._joints = None
        self._bodies_indexes = None
        self._world_points = None
        self._color_point_cloud_points = None
        self._depth_point_cloud_points = None
        self._body_point_cloud_points = None
        self._skeleton_point_cloud_points = None  #
        self._simultaneously_point_cloud_points = None  # stack all the points
        self._skeleton_colors = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0, 1, 1], [1, 0, 1]], dtype=np.float32)  # skeleton color pallet
        self._app = QtGui.QApplication([])  # Initialize app
        self._w = gl.GLViewWidget()  # Initialize view widget
        self._w.orbit(225, -30)
        self._w.showMaximized()  # show window maximized
        self._w.setWindowTitle('Kinect PointCloud')  # window title
        self._w.show()  
        self._scatter = None
        self._color = None
        self._t = None
        self._start = True
        self._start_gui = False
        self._dynamic_point_cloud = None  # Store the calculated point cloud points
        self._points_model = []
        #self.array_3d_model = np.array([np.zeros((217092, 3))])
        # check for multiple input flags or no input flags when using dynamic point cloud only
        if self._dynamic:
            if  any([self._depth_point_cloud and self._skeleton_point_cloud, self._body_index_cloud and self._skeleton_point_cloud]):
                self.init()  # Initialize the GL GUI

    def load_data(self):
        # initialize zeros points just for initialization
        self._dynamic_point_cloud = np.ndarray(shape=(2, 3), dtype=np.float32)
        # Initialize color and plot the scatter points
        self._color = np.zeros((len(self._dynamic_point_cloud), 4), dtype=np.float32)
        self._color[:, :] = 1
        self._scatter = gl.GLScatterPlotItem(pos=self._dynamic_point_cloud, size=self._size, color=self._color)  # create first scatter points
        self._w.addItem(self._scatter)  # add items

    def show(self):
        x = np.array([])
        y = np.array([])
        z = np.array([])

        for i in range(0, len(self.array_3d_model)):
            x = np.append(x, self.array_3d_model[i][:][:, 0])
            y = np.append(y, self.array_3d_model[i][:][:, 1])
            z = np.append(z, self.array_3d_model[i][:][:, 2])

        #print(len(x))

        #x, y, z = np.broadcast_arrays(x, y, z)


        # Do the plotting in a single call.
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x, y, z)
        plt.show()

    def update(self):
        self._opacity = 255
        color = False
        depth = True
        body = True
        skeleton = True
        simultaneously = True
        self._color_point_cloud = True if color == 1 else False
        self._simultaneously_point_cloud = True if simultaneously == 1 else False
        self._depth_point_cloud = True if depth == 1 else False
        self._body_index_cloud = True if body == 1 else False
        self._skeleton_point_cloud = True if skeleton == 1 else False

        # only for dynamic pointcloud
        if self._dynamic:
            self._color_point_cloud_points = self._dynamic_point_cloud

            # for depth point cloud
            if self._depth_point_cloud:
                self._world_points = mapper.depth_2_world(self._kinect, self._kinect._depth_frame_data, _CameraSpacePoint)
                self._world_points = ctypes.cast(self._world_points, ctypes.POINTER(ctypes.c_float))
                self._world_points = np.ctypeslib.as_array(self._world_points, shape=(self._kinect.depth_frame_desc.Height * self._kinect.depth_frame_desc.Width, 3))
                # store points
                self._dynamic_point_cloud = np.ndarray(shape=(len(self._world_points), 3), dtype=np.float32)
                self._dynamic_point_cloud[:, 0] = self._world_points[:, 0] * 1000
                self._dynamic_point_cloud[:, 1] = self._world_points[:, 2] * 1000
                self._dynamic_point_cloud[:, 2] = self._world_points[:, 1] * 1000
                # simultaneously point cloud
                if self._simultaneously_point_cloud:
                    self._depth_point_cloud_points = self._dynamic_point_cloud

            # for body index point cloud
            if self._body_index_cloud:
                self._dynamic_point_cloud = np.ndarray(shape=(2, 3), dtype=np.float32)

                # simultaneously point cloud
                if self._simultaneously_point_cloud:
                    self._body_point_cloud_points = self._dynamic_point_cloud

                # simultaneously point cloud
                if self._simultaneously_point_cloud:
                    self._skeleton_point_cloud_points = self._dynamic_point_cloud

            # for simultaneously point cloud stack the point arrays
            if self._simultaneously_point_cloud:
                self._simultaneously_point_cloud_points = np.ndarray(shape=(1,3), dtype=np.float32)
                if self._color_point_cloud:
                    self._simultaneously_point_cloud_points = np.vstack((self._simultaneously_point_cloud_points, self._color_point_cloud_points))
                if self._depth_point_cloud:
                    self._simultaneously_point_cloud_points = np.vstack((self._simultaneously_point_cloud_points, self._depth_point_cloud_points))
                if self._body_index_cloud:
                    self._simultaneously_point_cloud_points = np.vstack((self._simultaneously_point_cloud_points, self._body_point_cloud_points))
                if self._skeleton_point_cloud:
                    self._simultaneously_point_cloud_points = np.vstack((self._simultaneously_point_cloud_points, self._skeleton_point_cloud_points))
                # remove the first initialized array
                self._dynamic_point_cloud = self._simultaneously_point_cloud_points[1:,:]

        self._color = np.zeros((len(self._dynamic_point_cloud), 4), dtype=np.float32)
        self._color[:, 0] = self._red
        self._color[:, 1] = self._green
        self._color[:, 2] = self._blue
        self._color[:, 3] = self._opacity


        # update the pyqtgraph cloud

        # for i in self._dynamic_point_cloud:
        #      if i[1] >800:
        #          i[1] = 1000000000

        self.array_3d_model = np.array([np.zeros((217092, 3))])
        self.array_3d_model = np.append(self.array_3d_model, np.array([self._dynamic_point_cloud]), axis=0)
        self._points_model.append(self._dynamic_point_cloud)
        pcl.show()

        frame = Frame()
        self.get_frame(frame)
        self._scatter.setData(pos=self._dynamic_point_cloud, color=self._color, size=self._size)
        p7 = self._dynamic_point_cloud[108526]
        p4 = self._dynamic_point_cloud[108546]
        p13 = self._dynamic_point_cloud[108566]
        # N1 versor
        normaN1 = math.sqrt((p7[0] - p4[0]) ** 2 + (p7[1] - p4[1]) ** 2 + (p7[2] - p4[2]) ** 2)

        N1 = PyKinectV2._Joint()
        N1.x = ((p7[0] - p4[0]) / normaN1)
        N1.y = ((p7[1] - p4[1]) / normaN1)
        N1.z = ((p7[2] - p4[0]) / normaN1)


        # Tmp versor
        normaU = math.sqrt((p7[0] - p13[0]) ** 2 + (p7[1] - p13[1]) ** 2 + (p7[2] - p13[2]) ** 2)
        U = PyKinectV2._Joint()
        U.x = ((p7[0] - p13[0]) / normaU)
        U.y = ((p7[1] - p13[1]) / normaU)
        U.z = ((p7[2]- p13[2]) / normaU)

        # N3 versor
        N3 = PyKinectV2._Joint()
        N3.x = N1.y * U.z - N1.z * U.y
        N3.y = N1.z * U.x - N1.x * U.z
        N3.z = N1.x * U.y - N1.y * U.x

        normaN3 = math.sqrt(N3.x ** 2 + N3.y ** 2 + N3.z ** 2)
        N3.x = N3.x / normaN3
        N3.y = N3.y / normaN3
        N3.z = N3.z / normaN3

        # N2 versor
        N2 = PyKinectV2._Joint()
        N2.x = N3.y * N1.z - N3.z * N1.y
        N2.y = N3.z * N1.x - N3.x * N1.z
        N2.z = N3.x * N1.y - N3.y * N1.x

        matrR = np.array([[N1.x, N1.y, N1.z], [N2.x, N2.y, N2.z], [N3.x, N3.y, N3.z]])
        euler = self.rotationMatrixToEulerAngles(matrR)


        shoulder_pitch = euler[0] * 180 / np.pi
        shoulder_yaw = euler[1] * 180 / np.pi
        shoulder_roll = euler[2] * 180 / np.pi

        matrR = np.array([[N1.x, N1.y, N1.z, 0], [N2.x, N2.y, N2.z, 0], [N3.x, N3.y, N3.z, 0], [0, 0, 0, 1]])
        quat = self.rotationMatrixToQuaternion(matrR)

        print("roll:", shoulder_roll, "pitch: ", shoulder_pitch, "yaw: ", shoulder_yaw)
        print(quat)

    def init(self):
        self.load_data()  # load points for the first time
        self._t = QtCore.QTimer()  # initialize the Qui time
        self._t.timeout.connect(self.update)  # Initialize the update function
        self._t.start(10)  # import a delay

    def visualize(self):
        # start loop
        self._start = True
        while self._start:
            # check for interactive display and version
            if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
                # check to break loop
                if self._start_gui:
                    break
                # start app
                QtGui.QApplication.instance().exec_()
                self._start_gui = True
            else:
                self._start = False
        self._start = False
        pcl.show()
        cv2.destroyAllWindows()  # destroy track bar window and close application

    def get_frame(self, frame):
        self.acquireFrame()
        frame.ts = int(round(time.time() * 1000))

        self.frameNum += 1
        frame.frameRGB = self._frameRGB
        frame.frameDepth = self._frameDepth
        frame.frameDepthQuantized = self._frameDepthQuantized
        frame.frameSkeleton = self._frameSkeleton

        frame.frame_num = self.frameNum

        # get shoulder rotations (yaw, pitch and roll in Euler angles)
        euler, quat = self.get_shoulder_angles()
        if euler is None:
            return
        frame.shoulder_orientation_euler = euler
        frame.shoulder_orientation_quat = quat
        self.euler_to_quaternion(euler['roll']*np.pi/180, euler['pitch']*np.pi/180, euler['yaw']*np.pi/180)

    def get_shoulder_angles(self):

        # Find versors
        if self.joint_points3D is None or len(self.joint_points3D) == 0:
            return None, None

        p7 = self.joint_points3D[PyKinectV2.JointType_ShoulderRight]
        p4 = self.joint_points3D[PyKinectV2.JointType_ShoulderLeft]
        p13 = self.joint_points3D[PyKinectV2.JointType_SpineBase]

        # N1 versor
        normaN1 = math.sqrt((p7.x - p4.x)**2 + (p7.y - p4.y)**2 + (p7.z - p4.z)**2)
        N1 = PyKinectV2._Joint()
        N1.x = ((p7.x - p4.x) / normaN1)
        N1.y = ((p7.y - p4.y) / normaN1)
        N1.z = ((p7.z - p4.z) / normaN1)

        # Tmp versor
        normaU = math.sqrt((p7.x - p13.x)**2 + (p7.y - p13.y)**2 + (p7.z - p13.z)**2)
        U = PyKinectV2._Joint()
        U.x = ((p7.x - p13.x) / normaU)
        U.y = ((p7.y - p13.y) / normaU)
        U.z = ((p7.z - p13.z) / normaU)

        # N3 versor
        N3 = PyKinectV2._Joint()
        N3.x = N1.y * U.z - N1.z * U.y
        N3.y = N1.z * U.x - N1.x * U.z
        N3.z = N1.x * U.y - N1.y * U.x

        normaN3 = math.sqrt(N3.x**2 + N3.y**2 + N3.z**2)
        N3.x = N3.x / normaN3
        N3.y = N3.y / normaN3
        N3.z = N3.z / normaN3

        # N2 versor
        N2 = PyKinectV2._Joint()
        N2.x = N3.y * N1.z - N3.z * N1.y
        N2.y = N3.z * N1.x - N3.x * N1.z
        N2.z = N3.x * N1.y - N3.y * N1.x

        matrR = np.array([[N1.x,N1.y,N1.z],[N2.x,N2.y,N2.z],[N3.x,N3.y,N3.z]])
        euler = self.rotationMatrixToEulerAngles(matrR)
        shoulder_pitch = euler[0] * 180 / np.pi
        shoulder_yaw   = euler[1] * 180 / np.pi
        shoulder_roll  = euler[2] * 180 / np.pi
        print(matrR)
        matrR = np.array([[N1.x, N1.y, N1.z, 0], [N2.x, N2.y, N2.z, 0], [N3.x, N3.y, N3.z, 0], [0, 0, 0, 1]])
        quat = self.rotationMatrixToQuaternion(matrR)
        print(matrR)
        return {"roll": shoulder_roll, "pitch": shoulder_pitch, "yaw": shoulder_yaw}, quat

    # Checks if a matrix is a valid rotation matrix.
    def isRotationMatrix(self, R):
        Rt = np.transpose(R)
        shouldBeIdentity = np.dot(Rt, R)
        I = np.identity(3, dtype=R.dtype)
        n = np.linalg.norm(I - shouldBeIdentity)
        return n < 1e-6

    # Calculates rotation matrix to euler angles. The result is the same as MATLAB except the order of the euler angles ( x and z are swapped ).
    def rotationMatrixToEulerAngles(self, R):

        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

        singular = sy < 1e-6

        if not singular:
            x = math.atan2(R[2, 1], R[2, 2])
            y = math.atan2(-R[2, 0], sy)
            z = math.atan2(R[1, 0], R[0, 0])
        else:
            x = math.atan2(-R[1, 2], R[1, 1])
            y = math.atan2(-R[2, 0], sy)
            z = 0

        return np.array([x, y, z])


    def rotationMatrixToQuaternion(self, matrix, isprecise=True):

        M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
        if isprecise:
            q = np.empty((4,))
            t = np.trace(M)
            if t > M[3, 3]:
                q[0] = t
                q[3] = M[1, 0] - M[0, 1]
                q[2] = M[0, 2] - M[2, 0]
                q[1] = M[2, 1] - M[1, 2]
            else:
                i, j, k = 1, 2, 3
                if M[1, 1] > M[0, 0]:
                    i, j, k = 2, 3, 1
                if M[2, 2] > M[i, i]:
                    i, j, k = 3, 1, 2
                t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
                q[i] = t
                q[j] = M[i, j] + M[j, i]
                q[k] = M[k, i] + M[i, k]
                q[3] = M[k, j] - M[j, k]
            q *= 0.5 / math.sqrt(t * M[3, 3])
        else:
            m00 = M[0, 0]
            m01 = M[0, 1]
            m02 = M[0, 2]
            m10 = M[1, 0]
            m11 = M[1, 1]
            m12 = M[1, 2]
            m20 = M[2, 0]
            m21 = M[2, 1]
            m22 = M[2, 2]
            # symmetric matrix K
            K = np.array([[m00 - m11 - m22, 0.0, 0.0, 0.0],
                             [m01 + m10, m11 - m00 - m22, 0.0, 0.0],
                             [m02 + m20, m12 + m21, m22 - m00 - m11, 0.0],
                             [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22]])
            K /= 3.0
            # quaternion is eigenvector of K that corresponds to largest eigenvalue
            w, V = np.linalg.eigh(K)
            q = V[[3, 0, 1, 2], np.argmax(w)]
        if q[0] < 0.0:
            np.negative(q, q)
        return q

    def acquireFrame(self):

        if self._kinect.has_new_depth_frame():
            self.get_depth_frame()

        if self._kinect.has_new_color_frame():
            self.get_color_frame()

        if self._kinect.has_new_body_frame():
            self._bodies = self._kinect.get_last_body_frame()

        # --- draw skeletons to _frame_surface
        if self._bodies is not None:
            for i in range(0, self._kinect.max_body_count)[::-1]:
                body = self._bodies.bodies[i]
                if not body.is_tracked:
                    self.body_tracked = False
                    continue

                self.body_tracked = True

                joints = body.joints
                # convert joint coordinates to color space
                self.joint_points3D = self.body_joints_to_camera_space(joints)
                self.joint_points = self.body_joints_to_depth_space(joints)
                self.joint_points_RGB = self.body_joints_to_color_space(joints)
                self.joint_state = self.body_joints_state(joints)

if __name__ == "__main__":
    pcl = Cloud()
    pcl.visualize()
