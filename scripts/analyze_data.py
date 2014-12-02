#!/usr/bin/env python

"""
Analyze bag file data
"""

from collections import defaultdict

import rospy, rosbag, roslib
roslib.load_manifest('cv_bridge')
import cv_bridge

import numpy as np
import matplotlib.pyplot as plt
import cv2

import IPython

class AnalyzeData:
    def __init__(self, bag_name):
        bag = rosbag.Bag(bag_name, 'r')
        self.topics = defaultdict(list)
        
        for topic, msg, t in bag.read_messages():
            self.topics[topic].append((t, msg))
            
        self.analyze_forces = AnalyzeForces(self.topics['/pressure/l_gripper_motor'])
        self.analyze_images = AnalyzeImages(self.topics['/l_forearm_cam/image_rect_color/compressed'])
        
    def display_forces(self):
        self.analyze_forces.display()
        
        
    def display_images(self):
        self.analyze_images.display()
        
class AnalyzeForces:
    def __init__(self, force_msgs):
        """
        :param force_msgs: list of (time, msg)
        """
        start_time = force_msgs[0][0].to_sec()
        start_forces_l = np.array(force_msgs[0][1].l_finger_tip)
        start_forces_r = np.array(force_msgs[0][1].r_finger_tip)
        
        N = self.N = len(start_forces_l)
        T = self.T = len(force_msgs)
        
        self.times = [f[0].to_sec() - start_time for f in force_msgs]
        # each column is a time slice
        self.forces_l = np.zeros((N, T))
        self.forces_r = np.zeros((N, T))
        
        for t in xrange(T):
            self.forces_l[:,t] = force_msgs[t][1].l_finger_tip - start_forces_l
            self.forces_r[:,t] = force_msgs[t][1].r_finger_tip - start_forces_r
            
    def display(self):
        for i in xrange(self.N):
            plt.figure(1)
            plt.subplot(self.N, 1, i+1)
            if i == 0:
                plt.title('Left finger')
            plt.plot(self.times, self.forces_l[i,:], 'r')
            
            plt.figure(2)
            plt.subplot(self.N, 1, i+1)
            if i == 0:
                plt.title('Right finger')
            plt.plot(self.times, self.forces_r[i,:], 'b')
            
        plt.show(block=False)
        
class AnalyzeImages:
    def __init__(self, image_msgs):
        """
        :param image_msgs: list of (time, msg)
        """
        cv = cv_bridge.CvBridge()
        
        self.times = [msg[0].to_sec() for msg in image_msgs]
        #self.ims = [cv.imgmsg_to_cv(msg[1]) for msg in image_msgs]
        self.ims = [cv2.imdecode(np.fromstring(msg[1].data, np.uint8), cv2.CV_LOAD_IMAGE_COLOR) for msg in image_msgs]
        
        self.T = len(self.ims)

    def display(self, T=5):
        T = min(T, self.T)
        
        step = int(self.T/float(T))
        plt.subplot(T, 1, 1)
        ims_sub = self.ims[::step]
        for t, im in enumerate(ims_sub):
            plt.subplot(len(ims_sub), 1, t+1)
            if t == 0:
                plt.title('{0:.2f} seconds between images'.format(self.times[step] - self.times[0]))
            plt.imshow(im)
            
        
        plt.show(block=False)
        
        
#########
# TESTS #
#########

def test():
    rospy.init_node('analyze_data', anonymous=True)
    
    ad = AnalyzeData('../data/push_box.bag')
    
    ad.display_forces()
    #ad.display_images()
    
    print('Press enter to exit')
    raw_input()
        
if __name__ == '__main__':
    test()
