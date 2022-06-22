#!/usr/bin/python3.8

from time import time
from turtle import up
import rospy
import tf
from nav_msgs.msg import Odometry
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped, Point
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
import actionlib

import numpy as np
import math
from scipy.spatial.transform import Rotation as R

import numpy as np
import torch
from mants_sim_to_real.envs.Graph_Env import GraphHabitatEnv
from mants_sim_to_real.utils.config import get_config

from setArgs import tans_runner_init

reference_frame = ["/mkn2/odom", "/mkn5/odom"]
area = 114800
max_size = np.array((540,540))

class tansNode:
    def __init__(self, robots_list) -> None:
        self.robot_num = len(robots_list)
        self.robots_list = robots_list

        self.map_listener = [tf.TransformListener() for i in range(self.robot_num)]
        self.pose_listener = [tf.TransformListener() for i in range(self.robot_num)]
        self.goal_listener = [tf.TransformListener() for i in range(self.robot_num)]
        self.map = [None for i in range(self.robot_num)]
        self.map_corner = [None for i in range(self.robot_num)]
        self.current_pose = [None for i in range(self.robot_num)]

        self.global_goal = [None for i in range(self.robot_num)]
        self.distance_to_goal = [0 for i in range(self.robot_num)]
        self.new_goal = False
        
        self.explored_maps = [None for i in range(self.robot_num)]
        self.explored_maps_without_obstacles = [None for i in range(self.robot_num)]
        self.obstacle_maps = [None for i in range(self.robot_num)]
        self.map_poses = [None for i in range(self.robot_num)]
        self.poses = [None for i in range(self.robot_num)]
        self.ratios = [None for i in range(self.robot_num)]

        args = ['--n_rollout_threads', '1', '--hidden_size', '256', '--num_local_steps', '15', '--use_recurrent_policy', '--use_vo', 'ft_use_random', '--num_agents', '2', '--use_single', '--use_goal', '--grid_goal', '--use_grid_simple', '--grid_pos', '--grid_last_goal', '--cnn_use_transformer', '--use_share_cnn_model', '--agent_invariant', '--invariant_type', 'alter_attn', '--use_pos_embedding', '--use_id_embedding', '--multi_layer_cross_attn', '--add_grid_pos', '--use_self_attn', '--use_intra_attn', '--use_maxpool2d', '--cnn_layers_params', '32,3,1,1 64,3,1,1 128,3,1,1 64,3,1,1 32,3,2,1', '--model_dir', '/home/zzl/yxyWorkspace/src/mants_sim_to_real/data/model/tans']

        self.runner, self.num_local_steps = tans_runner_init(args)#init_graph_runner

        self.actoinclient = [actionlib.SimpleActionClient(robots_list[i]+'/move_base', MoveBaseAction) for i in range(self.robot_num)]
        self.goal_pub = [rospy.Publisher(robots_list[i]+"/goal", PoseStamped, queue_size=1) for i in range(self.robot_num)]
        
        self.start_time = rospy.get_time()
        self.last_step_time = self.start_time
        self.step = 0
        self.global_step = 0

        for i in range(self.robot_num):
            rospy.Subscriber(
                robots_list[i]+"/map_origin", OccupancyGrid, self.map_callback, callback_args=i, queue_size=1)
            rospy.Subscriber(
                robots_list[i]+"/odom", Odometry, self.pose_callback, callback_args=i, queue_size=1)
        
        rospy.Subscriber(
            robots_list[-1]+"/map_origin", OccupancyGrid, self.run_callback, queue_size=1)
        
        for i in range(self.robot_num):
            self.actoinclient[i].wait_for_server()
    
    def map_callback(self, data, robot_index) -> None:
        map_info = data.info
        shape = (map_info.height, map_info.width)
        timenow = rospy.Time(0)
    
        try:
            in_pose = PoseStamped()
            in_pose.header = data.header
            in_pose.pose.position.x, in_pose.pose.position.y, in_pose.pose.position.z = map_info.origin.position.x, map_info.origin.position.y, map_info.origin.position.z
            in_pose.pose.position.y += map_info.height * map_info.resolution
            in_pose.pose.orientation = map_info.origin.orientation

            self.map_listener[robot_index].waitForTransform(reference_frame[robot_index], data.header.frame_id, timenow, rospy.Duration(0.5))
            self.map_corner[robot_index] = self.map_listener[robot_index].transformPose(reference_frame[robot_index], in_pose)
            self.map_poses[robot_index] = (int(self.map_corner[robot_index].pose.position.x*20), int(self.map_corner[robot_index].pose.position.y*20))
            self.map[robot_index] = np.flip(np.asarray(data.data).reshape(shape), axis=0)

            temp = self.map[robot_index].copy()
            a = np.where(temp==-1)
            temp = temp.astype(float) / 100.0
            temp[np.where(temp==0)] = 1.0
            temp[a] = 0
            self.explored_maps[robot_index] = temp

            temp2 = self.map[robot_index].copy()
            c = np.where(temp2==-1)
            temp2 = temp2.astype(float) / 100.0
            a = np.where(temp2>0.2)
            b = np.where(temp2<=0.2)
            temp2[a] = 0
            temp2[b] = 1
            temp2[c] = 0
            self.explored_maps_without_obstacles[robot_index] = temp2

            temp3 = self.map[robot_index].copy()
            temp3[np.where(temp3==-1)] = 0
            temp3 = temp3.astype(float) / 100.0
            temp3[np.where(temp3<=0.4)] = 0
            self.obstacle_maps[robot_index] = temp3
        except:
            print("map listener fails")
    
    def pose_callback(self, data, robot_index) -> None:
        timenow = rospy.Time(0)

        try:
            in_pose = PoseStamped()
            in_pose.header = data.header
            in_pose.pose.position.x, in_pose.pose.position.y, in_pose.pose.position.z = data.pose.pose.position.x, data.pose.pose.position.y, data.pose.pose.position.z

            self.pose_listener[robot_index].waitForTransform(reference_frame[robot_index], data.header.frame_id, timenow, rospy.Duration(0.5))
            self.current_pose[robot_index] = self.pose_listener[robot_index].transformPose(reference_frame[robot_index], in_pose)
            self.poses[robot_index] = (int(self.current_pose[robot_index].pose.position.x*20), int(self.current_pose[robot_index].pose.position.y*20))
        except:
            print("pose listener fails")
    
    def run_callback(self, data):
        time_now = rospy.get_time()
        if time_now - self.last_step_time > 0.5:
            pos, ratio, explored_map, explored_map_no_obs, obstacle_map, left_corner = \
            self.poses, self.ratios, self.explored_maps, self.explored_maps_without_obstacles, self.obstacle_maps, self.map_poses 
            if self.step == 0:
                global_goal_position = self.runner.init_reset(max_size, pos, left_corner, obstacle_map, explored_map)#init_graph_runner
                self.global_goal = global_goal_position
                self.start_time = rospy.get_time()
                self.new_goal = True
            self.global_step = self.step // self.num_local_steps
            self.ratios = [0.4, 0.4]
            self.runner.get_pos(pos)

            for i in range(self.robot_num):
                self.distance_to_goal[i] = (self.global_goal[i][0] - self.poses[i][0])**2 + (self.global_goal[i][1] - self.poses[i][1])**2
                if self.distance_to_goal[i] <= 400:
                    self.new_goal = True

            if self.step % self.num_local_steps == self.num_local_steps-1:
                global_goal_position = self.runner.get_global_goal_position(pos, left_corner, obstacle_map, explored_map)
                print(global_goal_position)
                self.global_goal = global_goal_position
                for i in range(self.robot_num):
                    timenow = rospy.Time(0)
                    try:
                        goal_message = MoveBaseGoal()
                        goal_message.target_pose.header.frame_id = self.robots_list[i] + "/map"
                        d_x = global_goal_position[i][0] - self.poses[i][0]
                        d_y = global_goal_position[i][1] - self.poses[i][1]
                        if d_x>0:
                            if d_y>0:
                                euler = math.degrees(np.arctan(d_y/d_x))
                            else:
                                euler = math.degrees(-np.arctan(-d_y/d_x))
                        else:
                            if d_y>0:
                                euler = math.degrees(np.arctan(-d_x/d_y)) + 90
                            else:
                                euler = math.degrees(np.arctan(d_y/d_x)) + 180
                        euler -= 90 #transfermation between map and odom
                        orientation = R.from_euler('z', euler, degrees=True).as_quat()
                        in_pose = PoseStamped()
                        in_pose.header.frame_id = reference_frame[i]
                        in_pose.pose.position.x, in_pose.pose.position.y = global_goal_position[i][0]*0.05, global_goal_position[i][1]*0.05
                        self.goal_listener[i].waitForTransform(self.robots_list[i]+"/map", reference_frame[i], timenow, rospy.Duration(0.5))
                        goal_in_map = self.goal_listener[i].transformPose(self.robots_list[i]+"/map", in_pose)
                        goal_message.target_pose.pose.position = goal_in_map.pose.position
                        goal_message.target_pose.pose.orientation.x, goal_message.target_pose.pose.orientation.y, goal_message.target_pose.pose.orientation.z, goal_message.target_pose.pose.orientation.w = \
                            orientation[0], orientation[1], orientation[2], orientation[3]
                        self.actoinclient[i].send_goal(goal_message)

                        goal_marker = PoseStamped()
                        goal_marker.header.frame_id = self.robots_list[i] + "/map"
                        goal_marker.header.stamp = rospy.Time.now()
                        goal_marker.pose = goal_message.target_pose.pose
                        self.goal_pub[i].publish(goal_marker)
                    except:
                        print("goal tf fails")
                self.new_goal = False

            self.runner.render(obstacle_map, explored_map, pos, '/home/zzl/yxyWorkspace/src/toposim/test/figures/tans')
            self.step += 1
            self.last_step_time = rospy.get_time()
            print(self.last_step_time - self.start_time)