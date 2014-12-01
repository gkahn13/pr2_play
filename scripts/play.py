#!/usr/bin/env python

"""
Have the PR2 play and gather data
"""

import rospy, roslib
#roslib.load_manifest('pr2')
roslib.load_manifest('tfx')
import tfx

#from pr2_sim import arm
from pr2 import arm
import save_data

import IPython

def press_enter_to_continue():
    print('Press enter to continue')
    raw_input()

class Play:
    def __init__(self, material_name):
        self.material_name = material_name
        self.arm = arm.Arm('left')
    
    def touch(self):
        """
        Position the arm facing downwards and move down
        """
        home_pose = tfx.pose(tfx.pose([0.54, 0.2, 0.90], tfx.tb_angles(0,90,0), frame='base_link'))
        delta_z = -0.05
        speed = 0.02
        file_name = '../data/touch_{0}.bag'.format(self.material_name)
        print('Recording to file: {0}'.format(file_name))
        sd = save_data.SaveData(file_name, save_data.PR2_TOPICS_AND_TYPES)
        rospy.sleep(1)
        
        print('Going to home pose')
        #press_enter_to_continue()
        self.arm.go_to_pose(home_pose)
        
        print('Starting recording and moving by {0}'.format(delta_z))
        #press_enter_to_continue()
        sd.start()
        self.arm.go_to_pose(home_pose + [0, 0, delta_z], speed=speed)
        sd.stop()
        
        print('Stopping recording. Going to home pose')
        #press_enter_to_continue()
        self.arm.go_to_pose(home_pose)
    
    def push(self):
        """
        Push an object (with the material attached) across a surface
        """
        home_pose = tfx.pose([0.3, 0.48, 0.68], tfx.tb_angles(-90,0,0), frame='base_link')
        delta_y = -0.10
        speed = 0.02
        file_name = '../data/push_{0}.bag'.format(self.material_name)
        print('Recording to file: {0}'.format(file_name))
        sd = save_data.SaveData(file_name, save_data.PR2_TOPICS_AND_TYPES)
        
        print('Going to home pose')
        self.arm.go_to_pose(home_pose)
        
        print('Starting recording and moving by {0}'.format(delta_y))
        sd.start()
        self.arm.go_to_pose(home_pose + [0,delta_y,0], speed=speed)
        sd.stop()
        
        print('Stopping recording. Going to home pose')
        self.arm.go_to_pose(home_pose)

#########
# TESTS #
#########

def test_arm():
    a = arm.Arm('left')
    
    IPython.embed()

def test():
    play = Play('test')
    play.touch()    

if __name__ == '__main__':
    rospy.init_node('play', anonymous=True)
    rospy.sleep(0.5)
    test()
    #test_arm()
    
