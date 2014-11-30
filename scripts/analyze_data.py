#!/usr/bin/env python

"""
Analyze bag file data
"""

import rospy, rosbag
from collections import defaultdict

class AnalyzeData:
    def __init__(self, bag_name):
        bag = rosbag.Bag(bag_name)
        self.topics = defaultdict(list)
        
        for topic, msg, t in bag.read_messages():
            self.topics[topic].append((t, msg))
            
    ##########################
    # TODO: analysis methods #
    ##########################
        
        
#########
# TESTS #
#########

def test():
    rospy.init_node('analyze_data', anonymous=True)
    
    ad = AnalyzeData('../data/test.bag')
        
if __name__ == '__main__':
    test()