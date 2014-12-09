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
floor_materials = hards + big_cloths


class Play:
    def __init__(self, object_material, floor_material='None', pressure_threshold=10000):
        self.larm = arm.Arm('left', default_speed=0.08)
        self.rarm = arm.Arm('right', default_speed=0.08)
        
        self.pressure_sub = rospy.Subscriber('/pressure/l_gripper_motor', PressureState, self._pressure_callback)
        self.pressure_threshold = pressure_threshold
        
        self.reset(object_material, floor_material)
        
    def reset(self, object_material, floor_material='None'):
        self.object_material = object_material
        self.floor_material = floor_material
        
        self.last_forces_l = self.last_forces_r = None
        self.zero_forces_l = self.zero_forces_r = None
    
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
            
    def sound(self):
        """
        Position the gripper (grasping the rod) parallel to ground and move down
        BE SURE TO HAVE LAUNCH roslaunch audio_capture capture.launch
        """
        iters = 3 # number of times to hit
        home_pose = tfx.pose([0.48, -0.67, 0.85], tfx.tb_angles(0,0,0), frame='base_link')
        home_joints = [-1.49, 0.276231, -1.8, -1.43, 1.33627,-0.254, -25.1481]
        delta_pos = [0, 0, -0.10]
        speed = 0.25
        file = '../data/sound_{0}.bag'.format(self.object_material)
        
        #print('Recording to file: {0}'.format(file))
        #sd = save_data.SaveData(file, save_data.PR2_TOPICS_AND_TYPES)
        
        print('Going to home joints')
        self.rarm.go_to_joints(home_joints)
        
        print('Starting recording and moving by {0} for {1} times'.format(delta_pos, iters))
        #sd.start()
        for _ in xrange(iters):
            self.rarm.go_to_pose(home_pose + delta_pos, speed=speed, block=True)
            self.rarm.go_to_joints(home_joints)
        #sd.stop()
        
        
    def touch(self):
        """
        Position the gripper facing downwards and move down
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
        home_pose = tfx.pose([0.54, 0.2, 0.71], tfx.tb_angles(-90,0,0), frame='base_link')
        home_joints = [0.6857, 0.31154, 2.21, -1.062444, -0.33257,-1.212881, -0.81091]
        delta_pos = [0, -0.10, 0]
        speed = 0.02
        file = '../data/push_{0}_on_{1}.bag'.format(self.object_material, self.floor_material)
        
        self.execute_experiment(file, home_joints, home_pose, delta_pos, speed=speed)
        
    def execute_experiment(self, file, home_joints, home_pose, delta_pos, speed=0.02):
        self.larm.set_gripper(0.75*self.larm.min_grasp + 0.25*self.larm.max_grasp)
        print('Recording to file: {0}'.format(file))
        sd = save_data.SaveData(file, save_data.PR2_TOPICS_AND_TYPES)
        
        print('Going to home joints')
        self.larm.go_to_joints(home_joints)
        self.zero_pressure_sensors()
        
        print('Starting recording and moving by {0}'.format(delta_pos))
        sd.start()
        duration = self.larm.go_to_pose(home_pose + delta_pos, speed=speed, block=False)
        self.wait_action(duration)        
        sd.stop()
        
        print('Stopping recording. Going to home joints')
        self.larm.go_to_joints(home_joints)
        
        
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
    a = arm.Arm('right')
    
    p = a.get_pose()
    print('Pose : {0}'.format(p))
    home_pose = tfx.pose([0.48, -0.67, 0.85], tfx.tb_angles(0,0,0), frame='base_link')
    a.go_to_pose(home_pose)
    
    IPython.embed()

def test_home_pose():
    a = arm.Arm('left')
    
    # push home_pose
    home_pose = tfx.pose([0.54, 0.2, 0.71], tfx.tb_angles(-90,0,0), frame='base_link')
    home_joints = [0.6857, 0.31154, 2.21, -1.062444, -0.33257,-1.212881, -0.81091]
    a.go_to_joints(home_joints)
    
    IPython.embed()
    
def go_play_sound_all():
    play = Play('Does not matter', 'sound')
    for object_material in all_materials:
        if rospy.is_shutdown():
            break
        if object_material[0] != 'r':
            continue
        play.reset(object_material)
        print('Press enter to get sound of: {0}'.format(object_material))
        raw_input()
        print('(sound...)')
        play.sound()

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
    elif action == 'sound':
        print('Doing action ({0}) with material ({1})'.format(action, object_material))
        play.sound()

if __name__ == '__main__':
    rospy.init_node('play', anonymous=True)
    rospy.sleep(0.1)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('object_material')
    parser.add_argument('action', choices=('touch','push','sound'))
    parser.add_argument('floor_material', nargs='?', default='None')
    
    args = parser.parse_args(rospy.myargv()[1:])
    if args.object_material == 'all':
        if args.action == 'touch':
            go_play_touch_all()
        elif args.action == 'push':
            go_play_push_all()
        elif args.action == 'sound':
            go_play_sound_all()
    else:
        go_play(args.object_material, args.action, args.floor_material)
    
