#!/usr/bin/env python3
import rospy
import numpy as np
from nav_msgs.msg import OccupancyGrid
from obstacle_ransac.srv import localmap
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
import matplotlib.pyplot as plt

dummy_string = String()
rospy.init_node('repub_node', anonymous=True)
occupancy_pub = rospy.Publisher('/repub_grid', OccupancyGrid, queue_size=1)
rospy.wait_for_service('local_map')
lm = rospy.ServiceProxy('local_map', localmap)

time_since = rospy.Time.now()
curr_map = None
try:
    while not rospy.is_shutdown():
        curr_time = rospy.Time.now()
        try:
            if (curr_time-time_since).to_sec() > 0.1:
                # Half the map contents
                curr_map = lm(dummy_string)
                curr_map.map.data = np.array(curr_map.map.data)
                time_since = rospy.Time.now()
            
            occupancy_pub.publish(curr_map.map)
        except Exception as e:
            pass
                     

            
except rospy.ServiceException as e:
    print("Service call failed: %s"%e)