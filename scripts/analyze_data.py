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

import scipy as sci
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
        
        if 'push' in self.name:
            self.object_material = self.name.split('_')[1]
            self.floor_material = self.name.split('_')[-1]
        
            self.object_image = sci.misc.imread(data_folder + self.object_material + '.jpg')
            self.floor_img = sci.misc.imread(data_folder + self.floor_material + '.jpg')
        else:
            self.object_image = None
            self.floor_img = None
            
    ##########################
    # feature vector methods #
    ##########################
    
    @property
    def feature_vector(self):
        """
        Generate feature vector from data
        """
        return np.array([self.max_finger_tip_forces,
                        self.finger_tip_forces_energy,
                        self.distance_traveled,
                        self.max_joint_velocity,
                        self.max_joint_effort,
                        self.joint_effort_energy])
        
    @property
    def max_finger_tip_forces(self):
        return self.l_finger_tip.max(), self.r_finger_tip.max()

    @property
    def finger_tip_forces_energy(self):
        energy = 0
        for a in [self.l_finger_tip, self.r_finger_tip]:
            for col in xrange(a.shape[1]):
                energy += np.linalg.norm(np.convolve(a[:,col], [1,-1]))
        return energy

    @property
    def distance_traveled(self):
        return sum([self.poses[i].position.distance(self.poses[i+1].position) for i in xrange(len(self.poses)-1)])

    @property        
    def max_joint_velocity(self):
        return self.joint_velocities.max()
    
    @property
    def max_joint_effort(self):
        return self.joint_efforts.max()
        
    @property
    def joint_effort_energy(self):
        energy = 0
        for col in xrange(self.joint_efforts.shape[1]):
            energy += np.linalg.norm(np.convolve(self.joint_efforts[:,col], [1, -1]))
        return energy
                
        
    def __str__(self):
        """
        Print relevant statistics, for guidance
        """
        s = ''
        s += 'Max finger tip forces: {0}\n'.format(self.max_finger_tip_forces)
        s += 'Finger tip forces energy: {0:.3f}\n'.format(self.finger_tip_forces_energy)
        s += 'Distance traveled: {0:.3f}\n'.format(self.distance_traveled)
        s += 'Max joint velocity: {0:.3f}\n'.format(self.max_joint_velocity)
        s += 'Max joint effort: {0:.3f}\n'.format(self.max_joint_effort)
        s += 'Joint effort energy: {0:.3f}\n'.format(self.joint_effort_energy)
        return s
        
    ###################
    # display methods #
    ###################
        
    def display_forces(self, save_file=None):
        """
        Display fingertip pressure forces
        :param file to save figure to
        """
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

    def display_materials(self):
        """
        Display images of materials
        """
        if self.object_image is not None:
            plt.figure(1)
            plt.imshow(self.object_image)
            plt.title('Object material')
        if self.floor_img is not None:
            plt.figure(2)
            plt.imshow(self.floor_img)
            plt.title('Floor material')
            
        plt.show(block=False)

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
        
def print_push_files():
    for push_file in push_files:
        try:
            ad = AnalyzeData(data_folder + push_file + '.npz')
            print(push_file)
            print(str(ad)+'\n')
        except Exception as e:
            pass
            
            
if __name__ == '__main__':
    rospy.init_node('analyze_data', anonymous=True)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('npz_name')
    parser.add_argument('--save-all-display-forces', action='store_true')
    parser.add_argument('--print-push-files', action='store_true')

    args = parser.parse_args(rospy.myargv()[1:])
    
    if args.npz_name == 'all':
        if args.save_all_display_forces:
            save_all_display_forces()
        elif args.print_push_files:
            print_push_files()
    else:
        ad = AnalyzeData(data_folder + args.npz_name)
        #ad.display_forces()
        #ad.display_materials()
        print(ad)
        
    
    #print('Press enter to exit')
    #raw_input()
    

