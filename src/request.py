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
rospy.wait_for_service('localmap')
lm = rospy.ServiceProxy('localmap', localmap)

time_since = rospy.Time.now()
curr_map = None
try:
    while not rospy.is_shutdown():
        curr_time = rospy.Time.now()
        try:
            curr_map = lm(dummy_string)
            occupancy_msg = curr_map.map
            occupancy_msg.data = np.array(occupancy_msg.data)
            time_since = rospy.Time.now()
            
            occupancy_pub.publish(occupancy_msg)
            # plt.imshow(occupancy_msg.data.reshape(occupancy_msg.info.height,occupancy_msg.info.width))
            # plt.draw()
            # plt.pause(0.001)
            # plt.clf()
            # print("CLEARING")
        except Exception as e:
            print(e)
            pass
                     

            
except rospy.ServiceException as e:
    print("Service call failed: %s"%e)