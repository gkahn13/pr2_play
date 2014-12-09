#!/usr/bin/env python

"""
Extract info from bag files
"""

from collections import defaultdict
import itertools

import rospy, rosbag, roslib

import numpy as np

import argparse, os, binascii
import IPython

from pr2_sim import arm
from play import all_materials, object_materials, floor_materials

data_folder = '../data/'
push_files = [data_folder + 'push_{0}_on_{1}'.format(o, f) for o in object_materials for f in floor_materials]

joint_names =    ['fl_caster_rotation_joint',
                 'fl_caster_l_wheel_joint',
                 'fl_caster_r_wheel_joint',
                 'fr_caster_rotation_joint',
                 'fr_caster_l_wheel_joint',
                 'fr_caster_r_wheel_joint',
                 'bl_caster_rotation_joint',
                 'bl_caster_l_wheel_joint',
                 'bl_caster_r_wheel_joint',
                 'br_caster_rotation_joint',
                 'br_caster_l_wheel_joint',
                 'br_caster_r_wheel_joint',
                 'torso_lift_joint',
                 'head_pan_joint',
                 'head_tilt_joint',
                 'laser_tilt_mount_joint',
                 'r_upper_arm_roll_joint',
                 'r_shoulder_pan_joint',
                 'r_shoulder_lift_joint',
                 'r_forearm_roll_joint',
                 'r_elbow_flex_joint',
                 'r_wrist_flex_joint',
                 'r_wrist_roll_joint',
                 'r_gripper_joint',
                 'r_gripper_l_finger_joint',
                 'r_gripper_r_finger_joint',
                 'r_gripper_r_finger_tip_joint',
                 'r_gripper_l_finger_tip_joint',
                 'l_upper_arm_roll_joint',
                 'l_shoulder_pan_joint',
                 'l_shoulder_lift_joint',
                 'l_forearm_roll_joint',
                 'l_elbow_flex_joint',
                 'l_wrist_flex_joint',
                 'l_wrist_roll_joint',
                 'l_gripper_joint',
                 'l_gripper_l_finger_joint',
                 'l_gripper_r_finger_joint',
                 'l_gripper_r_finger_tip_joint',
                 'l_gripper_l_finger_tip_joint']


def extract_info_from_bag(bag_path, a, \
                          pressure_topic='/pressure/l_gripper_motor', \
                          joint_topic='/joint_states', \
                          audio_topic='/audio'):
    bag_name = bag_path.replace('.bag','')
    bag = rosbag.Bag(bag_path, 'r')
    topics = defaultdict(list)
    
    for topic, msg, t in bag.read_messages():
        topics[topic].append(msg)
                
    #########################
    # extract pressure info #
    #########################
    force_msgs = topics[pressure_topic]
    # each row is a timestamp
    l_finger_tip = np.array([f.l_finger_tip for f in force_msgs])
    r_finger_tip = np.array([f.r_finger_tip for f in force_msgs])
    t_pressure = np.array([(f.header.stamp - force_msgs[0].header.stamp).to_sec() for f in force_msgs])
    
    ######################
    # extract joint info #
    ######################
    joint_msgs = topics[joint_topic]
    # ensure ordering is always the same
    for j in joint_msgs:
        for j_name, name in zip(j.name, joint_names):
            assert(j_name == name)
    joint_positions = np.array([j.position for j in joint_msgs])
    joint_velocities = np.array([j.velocity for j in joint_msgs])
    joint_efforts = np.array([j.effort for j in joint_msgs])
    t_joint = np.array([(j.header.stamp - joint_msgs[0].header.stamp).to_sec() for j in joint_msgs])
    
    l_joint_indices = [i for i, n in enumerate(joint_names) if n in a.joint_names]
    
    joint_positions = joint_positions[:, l_joint_indices]
    joint_velocities = joint_velocities[:, l_joint_indices]
    joint_efforts = joint_efforts[:, l_joint_indices]
    
    #####################
    # extract pose info #
    #####################
    poses = np.zeros((len(joint_positions), 4, 4))
    for i, j in enumerate(joint_positions):
        a.set_joints(j)
        poses[i,:,:] = a.get_pose().matrix
        
    ###############################
    # extract audio info (if any) #
    ###############################
    audio = np.array([int(binascii.hexlify(d), 16) for m in topics[audio_topic] for d in m.data])

    if len(audio) > 0:
        audio_file = open(bag_name+'.mp3', 'w')
        for a in topics[audio_topic]:
            audio_file.write(''.join(a.data))
        audio_file.close()
    

    

    ################
    # save to .npz #
    ################
    np.savez(bag_name, l_finger_tip=l_finger_tip, r_finger_tip=r_finger_tip, t_pressure=t_pressure, \
        joint_positions=joint_positions, joint_velocities=joint_velocities, joint_efforts=joint_efforts, t_joint=t_joint, \
        poses=poses, audio=audio)
    
def extract_all():
    larm = arm.Arm('left', view=False)
    rarm = arm.Arm('right', view=False)
    
    for f in os.listdir(data_folder):
        f = data_folder + f
        if f.endswith('.bag'):
            print('Extracting info from: {0}'.format(f))

            extract_info_from_bag(f, larm if 'push' in f else rarm)
            
        
#########
# TESTS #
#########

if __name__ == '__main__':
    rospy.init_node('analyze_data', anonymous=True)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('bag_name')

    args = parser.parse_args(rospy.myargv()[1:])
    
    if args.bag_name == 'all':
        extract_all()
    else:
        extract_info_from_bag(data_folder + args.bag_name, arm.Arm('left', view=False))
    
    print('Press enter to exit')
    raw_input()
