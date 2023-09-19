#!/usr/bin/env python3
import numpy as np
from matplotlib import pyplot as plt
import rospy
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry, OccupancyGrid
from tf.transformations import euler_from_quaternion
from random import uniform,sample
import math
import rosparam
from scipy import ndimage

class OccupancyNode():
    def __init__(self):
        rospy.init_node('local_occupancy_node', anonymous=True)
        try:
            self.sub = rospy.Subscriber(rosparam.get_param('scan_topic'), LaserScan, self.scan_callback)
        except:
            self.sub = rospy.Subscriber('/car_1/scan', LaserScan, self.scan_callback)

        self.curr_pose = None
        self.odom_sub = rospy.Subscriber('/car_1/base/odom', Odometry, self.pose_callback)
        self.curr_heading = 0
        self.cartesian_points = None
        self.inlier_list = None
        self.map_resolution = rosparam.get_param('map_resolution')
        try:
            self.max_distance = rosparam.get_param('max_distance')
        except:
            self.max_distance = 5

        self.map_attributes = {
            'map_resolution': self.map_resolution,
            'width': 2*self.max_distance/self.map_resolution,
            'height': 2*self.max_distance/self.map_resolution,
        }

        self.occupancy_grid = np.zeros((int(self.map_attributes['height']),int(self.map_attributes['width'])),dtype=np.int8)
        self.occupancy_grid_pub = rospy.Publisher('/local_grid', OccupancyGrid, queue_size=1)

    def xy_to_cell_idx(self,coords):
        x = coords[0]
        y = coords[1]
        map_x = int((x+self.max_distance)/self.map_resolution)

        if map_x >= self.map_attributes['width']:
            map_x = int(self.map_attributes['width']-1)
        if map_x < 0:
            map_x = 0

        map_y = int(((y+self.max_distance)/self.map_resolution))

        if map_y >= self.map_attributes['height']:
            map_y = int(self.map_attributes['height']-1)
        if map_y < 0:
            map_y = 0

        return [map_y,map_x]
    
    def cell_idx_to_xy(self,map_idx):
        map_y = map_idx[0]
        map_x = map_idx[1]

        x = map_x*self.map_resolution-self.max_distance
        y = (self.map_attributes['height']-map_y)*self.map_resolution-self.max_distance
        return [x,y]

    def scan_callback(self, msg:LaserScan):
        if self.curr_pose is None:
            return
        theta = np.linspace(msg.angle_min, msg.angle_max, len(msg.ranges))

        valid_mask = [np.array(msg.ranges) < self.max_distance]
        polar = np.array(msg.ranges)[tuple(valid_mask)]
        theta = theta[tuple(valid_mask)]
        
        # polar = np.clip(np.array(msg.ranges),0,msg.range_max)

        # Convert to cartesian
        x = polar * np.cos(theta)
        y = polar * np.sin(theta)
        self.cartesian_points = np.vstack((x, y)).T

        # Convert to map coordinates
        map_idxs = np.array([self.xy_to_cell_idx(point) for point in self.cartesian_points])


        # self.clear_occupancy_grid()

        for point in map_idxs:
            self.occupancy_grid[point[0],point[1]] = 1

        # Apply dilation
        kernel = np.ones((3,3),np.uint8)
        dilated_occupancy_grid = ndimage.binary_dilation(self.occupancy_grid,structure=kernel).astype(self.occupancy_grid.dtype)
        # Publish occupancy grid

        dilated_occupancy_grid*=100

        occupancy_grid_msg = OccupancyGrid()
        occupancy_grid_msg.header.stamp = rospy.Time.now()
        occupancy_grid_msg.header.frame_id = 'car_1_base_link'
        occupancy_grid_msg.info.resolution = self.map_resolution
        occupancy_grid_msg.info.width = int(self.map_attributes['width'])
        occupancy_grid_msg.info.height = int(self.map_attributes['height'])
        occupancy_grid_msg.info.origin.position.x = -self.max_distance
        occupancy_grid_msg.info.origin.position.y = -self.max_distance
        occupancy_grid_msg.info.origin.position.z = 0
        occupancy_grid_msg.info.origin.orientation.x = 0
        occupancy_grid_msg.info.origin.orientation.y = 0
        occupancy_grid_msg.info.origin.orientation.z = 0
        occupancy_grid_msg.info.origin.orientation.w = 1
        occupancy_grid_msg.data = dilated_occupancy_grid.flatten().tolist()

        # print(type(occupancy_grid_msg.data))
        self.occupancy_grid_pub.publish(occupancy_grid_msg)

        

        self.clear_occupancy_grid()

    def clear_occupancy_grid(self):
        self.occupancy_grid = np.zeros((int(self.map_attributes['height']),int(self.map_attributes['width'])),dtype=np.int8)
        
    def pose_callback(self, msg):
        self.curr_pose = msg.pose.pose
        
        self.curr_heading = euler_from_quaternion([self.curr_pose.orientation.x, self.curr_pose.orientation.y, self.curr_pose.orientation.z, self.curr_pose.orientation.w])[2]
        # print(self.curr_heading)






    
if __name__ == '__main__':
    node = OccupancyNode()
    rate = rospy.Rate(20)
    while not rospy.is_shutdown():
        if node.cartesian_points is not None:
            try:
                rospy.loginfo(math.degrees(node.curr_heading))
            except Exception as e:
                print(e)
            
        else:
            print("EMPTY")
        rate.sleep()