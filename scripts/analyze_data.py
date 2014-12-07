#!/usr/bin/env python

"""
Analyze bag file data
"""

from collections import defaultdict

import rospy, rosbag, roslib
roslib.load_manifest('tfx')
import tfx
#roslib.load_manifest('cv_bridge')
#import cv_bridge

import numpy as np
import matplotlib.pyplot as plt
import cv2

import argparse
import IPython

from play import all_materials, object_materials, floor_materials

data_folder = '../data/'
image_folder = '../figs/'
push_files = ['push_{0}_on_{1}'.format(o, f) for o in object_materials for f in floor_materials]

class AnalyzeData:
    def __init__(self, npz_name):
        self.name = npz_name[:-4]
        npzfile = np.load(npz_name)
        
        self.l_finger_tip = npzfile['l_finger_tip']
        self.r_finger_tip = npzfile['r_finger_tip']
        self.t_pressure = npzfile['t_pressure']
        
        self.joint_positions = npzfile['joint_positions']
        self.joint_velocities = npzfile['joint_velocities']
        self.joint_efforts = npzfile['joint_efforts']
        self.t_joint = npzfile['t_joint']
        
        self.poses = [tfx.pose(p) for p in npzfile['poses']]
        
    ###################
    # display methods #
    ###################
        
    def display_forces(self, save_file=None):
        N = self.l_finger_tip.shape[1]
        
        f, axes = plt.subplots(N, 2, sharex=True, sharey=True)
        axes_left = axes[:,0]
        axes_right = axes[:,1]

        axes_left[0].set_title('Left finger')
        axes_right[0].set_title('Right finger')
    
        for i in xrange(N):
            axes_left[i].plot(self.t_pressure, self.l_finger_tip[:,i], 'r')
            axes_right[i].plot(self.t_pressure, self.r_finger_tip[:,i], 'b')
            
        f.subplots_adjust(hspace=0)
        f.suptitle(self.name)
            
        plt.show(block=False)
        
        if save_file is not None:
            plt.savefig(save_file)
        
        return f


#########
# TESTS #
#########

def save_all_display_forces():
    for push_file in push_files:
        try:
            ad = AnalyzeData(data_folder + push_file + '.npz')
            f = ad.display_forces(save_file=image_folder + push_file + '.jpg')
            print('Saved image for push_file: {0}'.format(push_file))
            plt.close()
        except:
            pass
        

if __name__ == '__main__':
    rospy.init_node('analyze_data', anonymous=True)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('npz_name')
    parser.add_argument('--save-all-display-forces', action='store_true')

    args = parser.parse_args(rospy.myargv()[1:])
    
    if args.npz_name == 'all':
        if args.save_all_display_forces:
            save_all_display_forces()
    else:
        ad = AnalyzeData(data_folder + args.npz_name)
        ad.display_forces()
    
    print('Press enter to exit')
    raw_input()
    

