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
from skimage import draw
from numba import jit
import time
import multiprocessing as mp

@jit(nopython=True)
def bres(p1,p2,w,h):
    (y0, x0) = p1
    (y1, x1) = p2

    steep = abs(y1 - y0) > abs(x1 - x0)
    if steep:
        x0, y0 = y0, x0  
        x1, y1 = y1, x1

    switched = False
    if x0 > x1:
        switched = True
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    if y0 < y1: 
        ystep = 1
    else:
        ystep = -1

    deltax = x1 - x0
    deltay = abs(y1 - y0)
    error = -deltax / 2
    y = y0

    line = []    
    for x in range(x0, x1 + 1):
        if x > w or x > h:
            continue
        if steep:
            line.append((y,x))
        else:
            line.append((x,y))
            

        error = error + deltay
        if error > 0:
            y = y + ystep
            error = error - deltax
    if switched:
        line.reverse()
    return line

class OccupancyNode():
    def __init__(self):
        
        
        self.curr_heading = 0
        self.cartesian_points = None
        self.inlier_list = None
        self.map_resolution = rosparam.get_param('map_resolution')

        self.custom_flag = rosparam.get_param('custom_size')
        if not self.custom_flag:
            try:
                self.max_distance = rosparam.get_param('max_distance')
            except:
                self.max_distance = 5

            self.map_attributes = {
                'map_resolution': self.map_resolution,
                'width': 2*self.max_distance/self.map_resolution,
                'height': 2*self.max_distance/self.map_resolution,
            }
            self.occupancy_grid = np.ones((int(self.map_attributes['height']),int(self.map_attributes['width'])),dtype=np.int8)

        else:
            top = rosparam.get_param('trbl')[0]
            right = rosparam.get_param('trbl')[1]
            bottom = rosparam.get_param('trbl')[2]
            left = rosparam.get_param('trbl')[3]

            tl_coord = [-left,top]
            br_coord = [right,-bottom]

            self.map_attributes = {
                'map_resolution': self.map_resolution,
                'width': int((br_coord[0]-tl_coord[0])/self.map_resolution),
                'height': int((tl_coord[1]-br_coord[1])/self.map_resolution),
                'tl_coord': tl_coord,
                'br_coord': br_coord,
                'origin': [-left,-bottom]
            }
            self.occupancy_grid = np.ones((int(self.map_attributes['height']),int(self.map_attributes['width'])),dtype=np.int8)
            
        print(self.map_attributes)

        self.occupancy_grid_pub = rospy.Publisher('/local_grid', OccupancyGrid, queue_size=0)

        self.occupancy_grid_msg = OccupancyGrid()
        
        self.curr_scan = None
        
        try:
            self.sub = rospy.Subscriber(rosparam.get_param('scan_topic'), LaserScan, self.scan_callback)
        except:
            self.sub = rospy.Subscriber('/car_1/scan', LaserScan, self.scan_callback)

        self.curr_pose = None
        self.odom_sub = rospy.Subscriber('/car_1/base/odom', Odometry, self.pose_callback)


    def xy_to_cell_idx(self,coords):

        if not self.custom_flag:
            x = coords[0]
            y = coords[1]
            map_x = int((x+self.max_distance)/self.map_resolution)


            map_y = int(((y+self.max_distance)/self.map_resolution))

        else:
            x = coords[0]
            y = coords[1]
            map_x = int((x-self.map_attributes['tl_coord'][0])/self.map_resolution)


            map_y = self.map_attributes['height']-int((self.map_attributes['tl_coord'][1]-y)/self.map_resolution)

    
        
        return [map_y,map_x]
    
    def bres(self,p1,p2):
        (y0, x0) = p1
        (y1, x1) = p2

        steep = abs(y1 - y0) > abs(x1 - x0)
        if steep:
            x0, y0 = y0, x0  
            x1, y1 = y1, x1

        switched = False
        if x0 > x1:
            switched = True
            x0, x1 = x1, x0
            y0, y1 = y1, y0

        if y0 < y1: 
            ystep = 1
        else:
            ystep = -1

        deltax = x1 - x0
        deltay = abs(y1 - y0)
        error = -deltax / 2
        y = y0

        line = []    

        for x in range(x0, x1 + 1):
            if steep:
                line.append((y,x))
            else:
                line.append((x,y))

            error = error + deltay
            if error > 0:
                y = y + ystep
                error = error - deltax
        if switched:
            line.reverse()
        return line
    
    def cell_idx_to_xy(self,map_idx):
        map_y = map_idx[0]
        map_x = map_idx[1]

        x = map_x*self.map_resolution-self.max_distance
        y = (self.map_attributes['height']-map_y)*self.map_resolution-self.max_distance
        return [x,y]

    def scan_callback(self, msg:LaserScan):
        if self.curr_scan is None:
            self.curr_scan = msg
    
        theta = np.linspace(msg.angle_min, msg.angle_max, len(msg.ranges))
        # valid_mask = [np.array(self.curr_scan.ranges) < self.max_distance]
        # polar = np.array(self.curr_scan.ranges)[tuple(valid_mask)]
        # theta = theta[tuple(valid_mask)]
        
        polar = np.clip(np.array(msg.ranges,dtype=np.float32),0,msg.range_max)

        # Convert to cartesian
        x = np.clip(polar * np.cos(theta),self.map_attributes['tl_coord'][0],msg.range_max)
        y = np.clip(polar * np.sin(theta),self.map_attributes['br_coord'][1],msg.range_max)

        self.cartesian_points = np.vstack((x, y)).T


        # Convert to map coordinates
        origin_px = self.xy_to_cell_idx([0,0])
        map_idxs = np.array([self.xy_to_cell_idx(point) for point in self.cartesian_points])

        # car_radius = int(rosparam.get_param('car_radius')/self.map_resolution)

        # for r in range(-car_radius,car_radius):
        #     for c in range(-car_radius,car_radius):
        #         if r**2+c**2 <= car_radius**2:
        #             self.occupancy_grid[origin_px[0]+r,origin_px[1]+c] = 0

        

        # self.clear_occupancy_grid()
        start = time.time()
        for point in map_idxs:
            try:
                # print(point)
                bres_line = bres(origin_px,point,self.map_attributes['width'],self.map_attributes['height'])
                for p in range(1,len(bres_line)-1):
                    if bres_line[p][0] < 0 or bres_line[p][1] < 0:
                        continue
                    if bres_line[p][0] >= self.occupancy_grid.shape[0] or bres_line[p][1] >= self.occupancy_grid.shape[1]:
                        continue
                    self.occupancy_grid[bres_line[p][1],bres_line[p][0]] = 0
                
                # bres_line = draw.line(origin_px[0],origin_px[1],point[0],point[1])
                # self.occupancy_grid[bres_line[0],bres_line[1]] = 0
                
                # if point[0] < 0 or point[1] < 0:
                #     continue
                # if point[0] >= self.occupancy_grid.shape[0] or point[1] >= self.occupancy_grid.shape[1]:
                #     continue
                # self.occupancy_grid[point[0],point[1]] = 0
            except:
                pass
        end = time.time()
        print((end-start))

        # Apply dilation
        dilated_occupancy_grid = self.occupancy_grid.copy()
        if rosparam.get_param('dilation'):
            kernel_size = rosparam.get_param('dilation_kernel')
            kernel = np.ones((kernel_size,kernel_size),np.uint8)
            # dilated_occupancy_grid = ndimage.median_filter(self.occupancy_grid,size=kernel_size)
            dilated_occupancy_grid = ndimage.minimum_filter(self.occupancy_grid,size=rosparam.get_param('minfilter_kernel'))
            dilated_occupancy_grid = ndimage.binary_dilation(dilated_occupancy_grid,structure=kernel).astype(self.occupancy_grid.dtype)

        dilated_occupancy_grid*=100


        # occupancy_grid_msg = OccupancyGrid()
        self.occupancy_grid_msg.header.stamp = rospy.Time.now()
        self.occupancy_grid_msg.header.frame_id = 'car_1_laser'
        self.occupancy_grid_msg.info.resolution = self.map_resolution

        self.occupancy_grid_msg.info.width = int(self.map_attributes['width'])
        self.occupancy_grid_msg.info.height = int(self.map_attributes['height'])
        if not self.custom_flag:
            self.occupancy_grid_msg.info.origin.position.x = -self.max_distance
            self.occupancy_grid_msg.info.origin.position.y = -self.max_distance

        else:
            self.occupancy_grid_msg.info.origin.position.x = self.map_attributes['origin'][0]
            self.occupancy_grid_msg.info.origin.position.y = self.map_attributes['origin'][1]

        self.occupancy_grid_msg.info.origin.position.z = 0
        self.occupancy_grid_msg.info.origin.orientation.x = 0
        self.occupancy_grid_msg.info.origin.orientation.y = 0
        self.occupancy_grid_msg.info.origin.orientation.z = 0
        self.occupancy_grid_msg.info.origin.orientation.w = 1
        self.occupancy_grid_msg.data = dilated_occupancy_grid.flatten().tolist()

        # print(type(occupancy_grid_msg.data))
        self.occupancy_grid_pub.publish(self.occupancy_grid_msg)

        

        self.clear_occupancy_grid()

    def clear_occupancy_grid(self):
        self.occupancy_grid = np.ones((int(self.map_attributes['height']),int(self.map_attributes['width'])),dtype=np.int8)
        
    def pose_callback(self, msg):
        self.curr_pose = msg.pose.pose
        
        self.curr_heading = euler_from_quaternion([self.curr_pose.orientation.x, self.curr_pose.orientation.y, self.curr_pose.orientation.z, self.curr_pose.orientation.w])[2]
        # print(self.curr_heading)
        





    
if __name__ == '__main__':
    rospy.init_node('local_occupancy_node', anonymous=True)
    node = OccupancyNode()
    print(node.xy_to_cell_idx([0,0]))
    rate = rospy.Rate(10)
    rospy.spin()