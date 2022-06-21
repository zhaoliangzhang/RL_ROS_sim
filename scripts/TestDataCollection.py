#!/usr/bin/python3.8

import rospy
import tf
from nav_msgs.msg import Odometry
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Twist, PoseStamped, Point

from PIL import Image
import numpy as np
from scipy.spatial.transform import Rotation as R

import pickle
import pdb

######### Pre-defined parameters ################
max_xize = (340, 500)
area = 114800


class DataCollectionNode:
    def __init__(self, robot_name) -> None:

        self.robot_name = robot_name
        self.map = None
        self.pose = None
        self.location = None
        self.ratio = 0
        self.tf_listener = tf.TransformListener()

        self.step = 0
        self.start_time = rospy.get_time() + 3

        self.data = {}
        self.explored_maps = []
        self.explored_maps_without_obstacles = []
        self.obstacle_maps = []
        self.map_poses = []
        self.poses = []
        self.ratios = []
        
        rospy.Subscriber(
            self.robot_name+"/odom", Odometry, self.pose_callback, queue_size=1)
        rospy.Subscriber(
            # self.robot_name+"/cartographer_discrete_map", OccupancyGrid, self.map_callback, queue_size=1)
            self.robot_name+"/map", OccupancyGrid, self.map_callback, queue_size=1)
    
    def pose_callback(self, data) -> None:
        time = rospy.get_time()
        if self.step == 0 and time>self.start_time:
            temp = self.map.copy()
            np.set_printoptions(threshold=np.inf)
            a = np.where(temp==-1)
            temp = temp.astype(float) / 100.0
            temp[np.where(temp<=0.2)] = 1.0
            temp[a] = 0
            self.explored_maps.append(temp)

            temp2 = self.map.copy()
            c = np.where(temp2==-1)
            temp2 = temp2.astype(float) / 100.0
            a = np.where(temp2>0.2)
            b = np.where(temp2<=0.2)
            temp2[a] = 0
            temp2[b] = 1
            temp2[c] = 0
            self.explored_maps_without_obstacles.append(temp2)

            temp3 = self.map.copy()
            temp3[np.where(temp3==-1)] = 0
            temp3 = temp3.astype(float) / 100.0
            temp3[np.where(temp3<=0.4)] = 0
            self.obstacle_maps.append(temp3)

            self.map_poses.append((int(self.location.pose.position.x*20), int(self.location.pose.position.y*20)))

            self.poses.append((int(data.pose.pose.position.x*20), int(data.pose.pose.position.y*20)))
            self.ratios.append(self.ratio)
            self.step += 1

        if time - self.start_time > self.step*2:
            self.poses.append((int(data.pose.pose.position.x*20), int(data.pose.pose.position.y*20)))
            self.ratios.append(self.ratio)
            print(self.step)
        
            if self.step>0 and self.step%15==0:
                temp = self.map.copy()
                a = np.where(temp==-1)
                temp = temp.astype(float) / 100.0
                temp[np.where(temp==0)] = 1.0
                temp[a] = 0
                self.explored_maps.append(temp)

                temp2 = self.map.copy()
                c = np.where(temp2==-1)
                temp2 = temp2.astype(float) / 100.0
                a = np.where(temp2>0.2)
                b = np.where(temp2<=0.2)
                temp2[a] = 0
                temp2[b] = 1
                temp2[c] = 0
                self.explored_maps_without_obstacles.append(temp2)

                temp3 = self.map.copy()
                temp3[np.where(temp3==-1)] = 0
                temp3 = temp3.astype(float) / 100.0
                temp3[np.where(temp3<=0.4)] = 0
                self.obstacle_maps.append(temp3)

                self.map_poses.append((int(self.location.pose.position.x*20), int(self.location.pose.position.y*20)))

                self.data = {}
                self.data["explored_maps"] = self.explored_maps
                self.data["explored_maps_without_obstacles"] = self.explored_maps_without_obstacles
                self.data["obstacle_maps"] = self.obstacle_maps
                self.data["map_poses"] = self.map_poses
                self.data["poses"] = self.poses
                self.data["ratios"] = self.ratios

                with open("/home/zzl/yxyWorkspace/debug/" + self.robot_name + "data.pkl", "wb") as tt:
                    pickle.dump(self.data, tt)

            self.step += 1

    def map_callback(self, data) -> None:
        map_info = data.info
        shape = (map_info.height, map_info.width)
        timenow = rospy.Time(0)
    
        try:
            in_pose = PoseStamped()
            in_pose.header = data.header
            in_pose.pose.position.x, in_pose.pose.position.y, in_pose.pose.position.z = map_info.origin.position.x, map_info.origin.position.y, map_info.origin.position.z
            in_pose.pose.position.y += map_info.height * map_info.resolution
            in_pose.pose.orientation = map_info.origin.orientation
            # print(map_info.height, map_info.resolution)
            # print("in", in_pose)
            # print("origin", map_info.origin)

            self.tf_listener.waitForTransform(reference_frame, data.header.frame_id, timenow, rospy.Duration(0.5))
            self.location = self.tf_listener.transformPose(reference_frame, in_pose)
            # print(self.location)
            self.map = np.flip(np.asarray(data.data).reshape(shape), axis=0)
            map_size = np.count_nonzero(self.map>=0)
            map_size -= np.count_nonzero(self.map==1)
            self.ratio = map_size / area
            # if self.robot_name=="mkn2":
            #     Image.fromarray(np.uint8(self.map*255)).convert("RGB").save("/home/zzl/yxyWorkspace/debug/map2", "png")
            # cv2.imwrite("/home/zzl/yxyWorkspace/debug/map.jpg", self.map)
        except:
            print("tf listener fails")

if __name__ == '__main__':
    rospy.init_node('data_collection')
    rn = rospy.get_param("~robot_name")
    reference_frame = rn+"/odom"

    node = DataCollectionNode(rn)

    rospy.spin()