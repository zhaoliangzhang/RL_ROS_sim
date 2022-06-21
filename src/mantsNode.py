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

# import sys
# sys.path.append("../..")

import numpy as np
import torch
from mants_sim_to_real.envs.Graph_Env import GraphHabitatEnv
from mants_sim_to_real.utils.config import get_config
from fake_sim import fakesim

reference_frame = ["/mkn2/odom", "/mkn5/odom"]
area = 114800
max_size = np.array((800,800))

def make_eval_env(all_args, run_dir):
    env = GraphHabitatEnv(args=all_args,run_dir=run_dir)
    return env

def parse_args(args, parser):
    parser.add_argument('--num_agents', type=int,
                        default=1, help="number of players")
    # visual params
    parser.add_argument("--render_merge", action='store_false', default=True,
                        help="by default, do not render the env during training. If set, start render. Note: something, the environment has internal render process which is not controlled by this hyperparam.")
    parser.add_argument("--visualize_input", action='store_true', default=False,
                        help="by default, do not render the env during training. If set, start render. Note: something, the environment has internal render process which is not controlled by this hyperparam.")
    # graph
    parser.add_argument('--build_graph', default=False, action='store_true')
    parser.add_argument('--add_ghost', default=False, action='store_true')
    parser.add_argument('--use_merge', default=False, action='store_true')
    parser.add_argument('--use_global_goal', default=False, action='store_true')
    parser.add_argument('--cut_ghost', default=False, action='store_true')
    parser.add_argument('--learn_to_build_graph', default=False, action='store_true')
    parser.add_argument('--use_mgnn', default=False, action='store_true')
    parser.add_argument('--dis_gap', default=2, type=int)
    parser.add_argument('--use_all_ghost_add', default=False, action='store_true')
    parser.add_argument('--ghost_node_size', default=12, type=int)
    parser.add_argument('--use_double_matching', default=False, action='store_true')
    parser.add_argument('--matching_type', type=str)
    parser.add_argument('--map_resolution', type=int, default=5)
    parser.add_argument('--graph_memory_size', default=100, type=int)
    parser.add_argument('--num_local_steps', type=int, default=25,
                    help="""Number of steps the local can
                        perform between each global instruction""")
    # graph
    parser.add_argument('--proj_frontier', action='store_true',
                        default=False, help="by default True, restrict goals to frontiers")
    parser.add_argument('--grid_pos', action='store_true',
                        default=False, help="by default True, use grid_pos")
    parser.add_argument('--agent_invariant', action='store_true',
                        default=False, help="by default True, ")
    parser.add_argument('--grid_goal', default=False, action='store_true')
    parser.add_argument('--use_goal', action='store_true',
                        default=False, help="by default True, use_goal")
    parser.add_argument('--use_local_single_map', action='store_true',
                        default=False, help="by default True, use_goal")
    parser.add_argument('--grid_last_goal', action='store_true',
                        default=False, help="by default True, use_goal")
    parser.add_argument('--add_grid_pos', action='store_true',
                        default=False, help="by default True, use_goal")
    parser.add_argument('--use_id_embedding', action='store_true',
                        default=False, help="by default True, use_goal")
    parser.add_argument('--use_pos_embedding', action='store_true',
                        default=False, help="by default True, use_goal")
    parser.add_argument('--use_intra_attn', action='store_true',
                        default=False, help="by default True, use_goal")
    parser.add_argument('--use_self_attn', action='store_true',
                        default=False, help="by default True, use_goal")
    parser.add_argument('--use_single', action='store_true',
                        default=False, help="by default True, use_goal")
    parser.add_argument('--use_grid_simple', action='store_true',
                        default=False, help="by default True, use_goal")
    parser.add_argument('--cnn_use_transformer', action='store_true',
                        default=False, help="by default True, use_goal")
    parser.add_argument('--use_share_cnn_model', action='store_true',
                        default=False, help="by default True, use_goal")
    parser.add_argument('--multi_layer_cross_attn', action='store_true',
                        default=False, help="by default True, use_goal")
    parser.add_argument('--invariant_type', type=str, default = "attn_sum", choices = ["attn_sum", "split_attn", "mean", "alter_attn"])             
    parser.add_argument('--attn_depth', default=2, type=int)
    parser.add_argument('--grid_size', default=8, type=int)
    parser.add_argument('--action_mask', default=False, action='store_true')


    # image retrieval
    all_args = parser.parse_known_args(args)[0]

    return all_args

def runner_init(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    # cuda 
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)
    
    run_dir = './test'
    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env init
    envs = make_eval_env(all_args, run_dir)
    eval_envs = None
    num_agents = all_args.num_agents

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir
    }

    # run experiments
    
    from mants_sim_to_real.runner.graph_habitat_runner import GraphHabitatRunner as Runner

    runner = Runner(config)
    
    return runner, all_args.num_local_steps

class mantsNode:
    def __init__(self, robots_list) -> None:
        self.robot_num = len(robots_list)
        self.robots_list = robots_list

        self.map_listener = [tf.TransformListener() for i in range(self.robot_num)]
        self.pose_listener = [tf.TransformListener() for i in range(self.robot_num)]
        self.goal_listener = [tf.TransformListener() for i in range(self.robot_num)]
        self.map = [None for i in range(self.robot_num)]
        self.map_corner = [None for i in range(self.robot_num)]
        self.current_pose = [None for i in range(self.robot_num)]
        
        self.explored_maps = [None for i in range(self.robot_num)]
        self.explored_maps_without_obstacles = [None for i in range(self.robot_num)]
        self.obstacle_maps = [None for i in range(self.robot_num)]
        self.map_poses = [None for i in range(self.robot_num)]
        self.poses = [None for i in range(self.robot_num)]
        self.ratios = [None for i in range(self.robot_num)]

        args = ['--n_rollout_threads', '1', '--ghost_node_size', '24', '--use_all_ghost_add', '--learn_to_build_graph', '--dis_gap', '4', '--graph_memory_size', '100', '--build_graph', '--use_merge', '--add_ghost', '--feature_dim', '512', '--hidden_size', '256', '--use_mgnn', '--use_global_goal', '--cut_ghost', '--num_local_steps', '6', '--use_recurrent_policy', '--num_agents', '2', '--model_dir', '/home/zzl/yxyWorkspace/src/mants_sim_to_real/data/model']
        self.runner, self.num_local_steps = runner_init(args)#init_graph_runner

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
            robots_list[0]+"/map_origin", OccupancyGrid, self.run_callback, queue_size=1)
        
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
                global_goal_position = self.runner.init_reset(pos, max_size, obstacle_map,left_corner)#init_graph_runner
            self.global_step = self.step // self.num_local_steps
            self.ratios = [0.4, 0.4]
            if self.step % self.num_local_steps == self.num_local_steps - 1:
                update = True
                infos = self.runner.build_graph(pos, left_corner, ratio, explored_map, explored_map_no_obs, obstacle_map, update)
            else:
                update = False
                infos = self.runner.build_graph(pos, left_corner, update = update)
            if self.step % self.num_local_steps == 0:
                global_goal_position = self.runner.get_global_goal(obstacle_map, self.global_step, infos)
                print(global_goal_position)
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

            self.runner.render(self.obstacle_maps, self.explored_maps, self.poses, '/home/zzl/yxyWorkspace/src/toposim/test/figures')
            self.step += 1
            self.last_step_time = rospy.get_time()

if __name__ == "__main__":
    rospy.init_node('topo_explore_runner')
    robots_list = ["mkn2", "mkn5"]
    # robots_list = ["mkn2"]

    node = mantsNode(robots_list)

    rospy.spin()