#!/usr/bin/env python

"""
Have the PR2 play and gather data
"""

import rospy, roslib
roslib.load_manifest('tfx')
roslib.load_manifest('pr2_msgs')
import tfx
from pr2_msgs.msg import PressureState

from pr2 import arm
from utils import utils
import save_data

import numpy as np
import argparse
import IPython


class Play:
    def __init__(self, material_name):
        self.material_name = material_name
        
        self.arm = arm.Arm('left', default_speed=0.08)
        self.arm.set_gripper(0.75*self.arm.min_grasp + 0.25*self.arm.max_grasp)
        
        self.pressure_sub = rospy.Subscriber('/pressure/l_gripper_motor', PressureState, self._pressure_callback)
        self.last_forces_l = self.last_forces_r = None
        self.zero_forces_l = self.zero_forces_r = None
        self.pressure_threshold = 5000
    
    def _pressure_callback(self, msg):
        self.last_forces_l = np.array(msg.l_finger_tip)
        self.last_forces_r = np.array(msg.r_finger_tip)
        
    def zero_pressure_sensors(self):
        while not rospy.is_shutdown() and (self.last_forces_l is None or self.last_forces_r is None):
            print('Waiting for pressure reading in order to zero...')
            rospy.sleep(1)
        self.zero_forces_l = self.last_forces_l
        self.zero_forces_r = self.last_forces_r
    
    def is_pressure_exceeded(self):
        return (self.last_forces_l - self.zero_forces_l).max() > self.pressure_threshold or \
            (self.last_forces_r - self.zero_forces_r).max() > self.pressure_threshold
    
    def touch(self):
        """
        Position the arm facing downwards and move down
        """
        home_pose = tfx.pose(tfx.pose([0.54, 0.2, 0.85], tfx.tb_angles(0,90,0), frame='base_link'))
        home_joints = [0.57, 0.1233, 1.288, -1.58564, 1.695, -1.85322, 14.727]
        delta_pos = [0, 0, -0.10]
        speed = 0.02
        file_name = '../data/touch_{0}.bag'.format(self.material_name)

        self.execute_experiment(file_name, home_joints, home_pose, delta_pos, speed=speed)
    
    def push(self):
        """
        Push an object (with the material attached) across a surface
        """
        home_pose = tfx.pose([0.54, 0.2, 0.8], tfx.tb_angles(-90,0,0), frame='base_link')
        home_joints = [0.7233, 0.118, 2.20911, -1.27236, -0.439,-1.13, 18.00]
        delta_pos = [0, -0.20, 0]
        speed = 0.02
        file_name = '../data/push_{0}.bag'.format(self.material_name)
        
        self.execute_experiment(file_name, home_joints, home_pose, delta_pos, speed=speed)
        
    def execute_experiment(self, file_name, home_joints, home_pose, delta_pos, speed=0.02):
        print('Recording to file: {0}'.format(file_name))
        sd = save_data.SaveData(file_name, save_data.PR2_TOPICS_AND_TYPES)
        
        print('Going to home joints')
        self.arm.go_to_joints(home_joints)
        self.zero_pressure_sensors()
        
        print('Starting recording and moving by {0}'.format(delta_pos))
        sd.start()
        duration = self.arm.go_to_pose(home_pose + delta_pos, speed=speed, block=False)
        self.wait_action(duration)        
        sd.stop()
        
        print('Stopping recording. Going to home joints')
        self.arm.go_to_joints(home_joints)
        
        
    def wait_action(self, duration):
        timeout = utils.Timeout(duration)
        timeout.start()
        while not rospy.is_shutdown():
            if timeout.has_timed_out():
                print('Action complete')
                break
            if self.is_pressure_exceeded():
                print('Pressure exceeded')
                break
            rospy.sleep(0.1)
        
        
    

#########
# TESTS #
#########

def test_arm():
    a = arm.Arm('left')
    
    IPython.embed()

    
def go_play(material_name, action):
    print('Doing action ({0}) with material ({1})'.format(action, material_name))
    play = Play(material_name)
    
    if action == 'touch':
        play.touch()
    elif action == 'push':
        play.push()

if __name__ == '__main__':
    rospy.init_node('play', anonymous=True)
    rospy.sleep(0.1)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('material_name')
    parser.add_argument('action', choices=('touch','push'))
    
    args = parser.parse_args(rospy.myargv()[1:])
    go_play(args.material_name, args.action)
    
    #test_arm()
    
