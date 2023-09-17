import numpy as np
from matplotlib import pyplot as plt
import rospy
from sklearn import linear_model
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion
from random import uniform,sample
import math
import time

def get_lines(curr_in_points):
    pass


class RansacNode():
    def __init__(self):
        rospy.init_node('ransac_node', anonymous=True)
        self.sub = rospy.Subscriber('/car_1/scan', LaserScan, self.scan_callback)
        self.curr_pose = None
        self.odom_sub = rospy.Subscriber('/car_1/base/odom', Odometry, self.pose_callback)
        self.curr_heading = 0
        self.cartesian_points = None
        self.inlier_list = None

    def scan_callback(self, msg):
        if self.curr_pose is None:
            return
        self.inlier_list = []
        valid_mask = [np.array(msg.ranges) < 5]
        polar = np.array(msg.ranges)[tuple(valid_mask)]
        theta = np.linspace(msg.angle_min, msg.angle_max, len(msg.ranges))
        theta = theta[tuple(valid_mask)]
        resolution = msg.angle_increment

        # Convert to cartesian
        x = self.curr_pose.position.x + np.round(polar * np.cos(self.curr_heading+theta),2)
        y = self.curr_pose.position.y + np.round(polar * np.sin(self.curr_heading+theta),2)
        self.cartesian_points = np.vstack((x, y)).T

        
    def pose_callback(self, msg):
        self.curr_pose = msg.pose.pose
        
        self.curr_heading = euler_from_quaternion([self.curr_pose.orientation.x, self.curr_pose.orientation.y, self.curr_pose.orientation.z, self.curr_pose.orientation.w])[2]
        # print(self.curr_heading)

    def compute_lines_sk(self, points):
        remaining_points = points.copy()
        original_points_count = len(remaining_points)
        min_ratio = 0.01
        trials = 5
        self.inlier_list = []
        # while len(self.inlier_list) < 4:
        final_inliers = []
        for i in range(5):
            if len(remaining_points) < original_points_count * min_ratio:
                return final_inliers
            X = remaining_points[:, 0].reshape(-1, 1)
            y = remaining_points[:, 1].reshape(-1, 1)
            

            # Robustly fit linear model with RANSAC algorithm
            ransac = linear_model.RANSACRegressor(residual_threshold=1)
            ransac.fit(X, y)
            inlier_mask = ransac.inlier_mask_
            
            inliers = remaining_points[inlier_mask]

            # Get 2 points furthest apart from current inliers

            dists = np.linalg.norm(inliers - inliers[:, None], axis=-1)
            max_dist = np.max(dists)
            max_idx = np.where(dists == max_dist)
            inliers = [list(inliers[max_idx[0][0]]), list(inliers[max_idx[1][0]])]
            
        
            final_inliers.append(inliers)

            # delete inliers from remaining points

            remaining_points = remaining_points[~inlier_mask]

        return final_inliers
    
    def distance_to_line(self,x0,y0,x1,y1,x2,y2):
        a = y2-y1
        b = x1-x2
        c = y1*x2-x1*y2
        dist = abs(a*x0+b*y0+c)/(math.sqrt((a)**2+(b)**2))
        # dist = abs((x2-x1)*(y1-y0)-(x1-x0)*(y2-y1))/(math.sqrt((x2-x1)**2+(y2-y1)**2))
        return dist

    def eucl(self,x1,y1,x2,y2):
        return math.sqrt((x1-x2)**2+(y1-y2)**2)

        
    def ransac(self,points):
        l_points = points.tolist()
        total_points = len(l_points)
        trials = 20
        inlier_threshold = 0.1
        min_points = 95/100
        # vals = {'pivot1':[],'pivot2':[],'inliersx':[],'inliersy':[],'count':0}
        binned_points = 0
        line_list = []
        for t1 in range(trials):
            # print(len(points))
            max_sub = 0
            vals = {'pivot1':[],'pivot2':[],'inliersx':[],'inliersy':[],'count':0}
            for t2 in range(trials):
                trial_sub = 0
                pivots = sample(l_points,2)
                pivot_1 = pivots[0]
                pivot_2 = pivots[1]
                inliersx = []
                inliersy = []
                for i in range(len(l_points)):
                    if(self.distance_to_line(l_points[i][0],l_points[i][1],pivot_1[0],pivot_1[1],pivot_2[0],pivot_2[1])<inlier_threshold):
                        trial_sub+=1
                        inliersx.append(l_points[i][0])
                        inliersy.append(l_points[i][1])

                if trial_sub>max_sub and trial_sub > 0.1*len(l_points):
                    max_sub=trial_sub
                    vals['inliersx']=inliersx
                    vals['inliersy']=inliersy
                    vals['pivot1']=pivot_1
                    vals['pivot2']=pivot_2

            for remo in range(len(vals['inliersx'])):

                l_points.remove([vals['inliersx'][remo],vals['inliersy'][remo]])
            line_list.append(vals)
            binned_points+=max_sub
            if(binned_points>min_points*total_points):
                break
            
        points_list = []
        for line_vals in line_list:
            new_points = []
            p1 = []
            p2 = []
            for j in range(len(line_vals['inliersx'])):
                new_points.append([line_vals['inliersx'][j],line_vals['inliersy'][j]])
            max_dist = 0
            p1 = new_points[0]
            p2 = new_points[-1]
            # for _ in range(trials):
                
            #     pivots = sample(new_points,2)
            #     pivot_1 = pivots[0]
            #     pivot_2 = pivots[1]
            #     test_dist = eucl(pivot_1[0],pivot_1[1],pivot_2[0],pivot_2[1])
            #     if(test_dist>max_dist):
            #         max_dist=test_dist
            #         p1 = pivot_1
            #         p2 = pivot_2

            points_list.append([p1,p2])

        return points_list



    
if __name__ == '__main__':
    node = RansacNode()
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        
        if node.cartesian_points is not None:
            try:
                start = time.time()
                lines = node.ransac(node.cartesian_points)
                print("CUSTOM:",time.time()-start)
                print(lines)
                
                start = time.time()
                lines_sk = node.compute_lines_sk(node.cartesian_points)
                print("SKLEARN:",time.time()-start)
                print(lines_sk)
                print('\n')
            except Exception as e:
                print(e)

        else:
            print("EMPTY")
        rate.sleep()