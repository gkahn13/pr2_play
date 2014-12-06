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

""" Names of materials (match with JPG images) """
big_cloths = ['bc{0}'.format(i) for i in xrange(1,6)]
hards = ['h{0}'.format(i) for i in xrange(1,6)]
rugs = ['r{0}'.format(i) for i in xrange(1,7)]
small_cloths = ['sc{0}'.format(i) for i in xrange(1,8)]

""" Name lists used for 'all' Play tests """
all_materials = big_cloths + hards + rugs + small_cloths
object_materials = small_cloths + rugs
floor_materials = big_cloths + hards


class Play:
    def __init__(self, object_material, floor_material='None', pressure_threshold=10000):
        self.arm = arm.Arm('left', default_speed=0.08)
        
        self.pressure_sub = rospy.Subscriber('/pressure/l_gripper_motor', PressureState, self._pressure_callback)
        self.pressure_threshold = pressure_threshold
        
        self.reset(object_material, floor_material)
        
    def reset(self, object_material, floor_material='None'):
        self.object_material = object_material
        self.floor_material = floor_material
        
        self.last_forces_l = self.last_forces_r = None
        self.zero_forces_l = self.zero_forces_r = None
        
        self.arm.set_gripper(0.75*self.arm.min_grasp + 0.25*self.arm.max_grasp)
    
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
        file = '../data/touch_{0}.bag'.format(self.object_material)

        self.execute_experiment(file, home_joints, home_pose, delta_pos, speed=speed)
    
    def push(self):
        """
        Push an object (with the material attached) across a surface
        """
        home_pose = tfx.pose([0.54, 0.2, 0.75], tfx.tb_angles(-90,0,0), frame='base_link')
        home_joints = [0.697, 0.27, 2.21, -1.1163, -0.35143,-1.19, -0.82]
        delta_pos = [0, -0.10, 0]
        speed = 0.02
        file = '../data/push_{0}_on_{1}.bag'.format(self.object_material, self.floor_material)
        
        self.execute_experiment(file, home_joints, home_pose, delta_pos, speed=speed)
        
    def execute_experiment(self, file, home_joints, home_pose, delta_pos, speed=0.02):
        print('Recording to file: {0}'.format(file))
        sd = save_data.SaveData(file, save_data.PR2_TOPICS_AND_TYPES)
        
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

def test_home_pose():
    a = arm.Arm('left')
    
    # push home_pose
    home_pose = tfx.pose([0.54, 0.2, 0.73], tfx.tb_angles(-90,0,0), frame='base_link')
    a.go_to_pose(home_pose)
    
    IPython.embed()

def go_play_touch_all():
    play = Play('Does not matter', 'touch')
    for object_material in all_materials:
        play.reset(object_material)
        print('Press enter to touch: {0}'.format(object_material))
        raw_input()
        print('Now touching {0}...'.format(object_material))
        play.touch()
    
def go_play_push_all():
    play = Play('Does not matter', 'push')
    for floor_material in floor_materials:
        print('########################')
        print('floor_material: {0}'.format(floor_material))
        print('########################')
        for object_material in object_materials:
            play.reset(object_material, floor_material)
            print('On floor_material: {0} -- pushing {1}'.format(floor_material, object_material))
            raw_input()
            print('(pushing...)')
            play.push()
            
    
def go_play(object_material, action, floor_material='None'):
    play = Play(object_material, floor_material=floor_material)
    
    if action == 'touch':
        print('Doing action ({0}) with material ({1})'.format(action, object_material))
        play.touch()
    elif action == 'push':
        print('Doing action ({0}) with material ({1}) on material ({2})'.format(action, object_material, floor_material))
        play.push()

if __name__ == '__main__':
    rospy.init_node('play', anonymous=True)
    rospy.sleep(0.1)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('object_material')
    parser.add_argument('action', choices=('touch','push'))
    parser.add_argument('floor_material', nargs='?', default='None')
    
    args = parser.parse_args(rospy.myargv()[1:])
    if args.object_material == 'all':
        if args.action == 'touch':
            go_play_touch_all()
        elif args.action == 'push':
            go_play_push_all()
    else:
        go_play(args.object_material, args.action, args.floor_material)
    
