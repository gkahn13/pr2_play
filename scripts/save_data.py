#!/usr/bin/env python

"""
Saves specifics topics to bag file
"""

import rospy, roslib, rosbag
roslib.load_manifest('pr2_msgs')

from sensor_msgs.msg import JointState, Image
from pr2_msgs.msg import PressureState

PR2_TOPICS_AND_TYPES = [('/joint_states', JointState),
                        ('/pressure/l_gripper_motor', PressureState),
                        ('/l_forearm_cam/image_rect_color', Image)]

class SaveData:
    def __init__(self, bag_name, topics_and_types):
        """
        :param bag_name: name of bag file to save to
        :param topics_and_types: list of tuples (topic, type) to subscribe to
        """
        self.bag = rosbag.Bag(bag_name, 'w')
        self.is_recording = False
        
        self.subs = list()
        for topic, type in topics_and_types:
            print('Subscribing to {0} of type {1}'.format(topic, str(type)))
            self.subs.append(rospy.Subscriber(topic, type, self._callback_wrapper(topic)))
    
    def start(self):
        self.is_recording = True
    
    def pause(self):
        self.is_recording = False
    
    def stop(self):
        self.pause()
        self.bag.close()
        
    def _callback_wrapper(self, topic):
        def _callback(data):
            if self.is_recording:
                self.bag.write(topic, data)
        return _callback

#########
# TESTS #
#########

def test():
    rospy.init_node('save_data', anonymous=True)
    
    topics_and_types = PR2_TOPICS_AND_TYPES
    sd = SaveData('../data/test.bag', topics_and_types)
    rospy.sleep(0.1)
    
    print('Recording...')
    sd.start()
    rospy.sleep(0.5)
    sd.stop()
    print('Saved!')
    

if __name__ == '__main__':
    test()