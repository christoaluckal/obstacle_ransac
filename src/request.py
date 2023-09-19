import rospy
import numpy as np
from nav_msgs.msg import OccupancyGrid
from obstacle_ransac.srv import localmap
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String

dummy_string = String()

rospy.wait_for_service('local_map')
try:
    lm = rospy.ServiceProxy('local_map', localmap)
    resp1 = lm(dummy_string)
    print(resp1)
except rospy.ServiceException as e:
    print("Service call failed: %s"%e)