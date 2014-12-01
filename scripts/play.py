#!/usr/bin/env python

"""
Have the PR2 play and gather data
"""

import rospy, roslib
#roslib.load_manifest('pr2')

#from pr2_sim import arm
from pr2 import arm
import save_data

import IPython

class Play:
    def __init__(self, material_name):
        self.material_name = material_name
        self.arm = arm.Arm('left')
    
    def touch(self):
        """
        Position the arm facing downwards and move down
        """
        home_pose = tfx.pose(tfx.pose([0.3, 0.68, 0.68], tfx.tb_angles(0,90,0), frame='base_link'))
        delta_z = -0.05
        speed = 0.02
        file_name = '../data/touch_{0}.bag'.format(self.material_name)
        print('Recording to file: {0}'.format(file_name))
        sd = save_data.SaveData(file_name, save_data.PR2_TOPICS_AND_TYPES)
        
        print('Going to home pose')
        self.arm.go_to_pose(home_pose)
        
        print('Starting recording and moving by {0}'.format(delta_z))
        sd.start()
        self.arm.go_to_pose(home_pose + [0, 0, delta_z], speed=speed)
        sd.stop()
        
        print('Stopping recording. Going to home pose')
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

def test():
    rospy.init_node('play', anonymous=True)
    a = arm.Arm('left')
    
    IPython.embed()

if __name__ == '__main__':
    test()