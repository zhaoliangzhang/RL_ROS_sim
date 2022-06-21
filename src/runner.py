#!/usr/bin/python3.8

import rospy
from mantsNode import mantsNode



if __name__ == "__main__":
    rospy.init_node('topo_explore_runner')
    robots_list = ["mkn2", "mkn5"]
    # robots_list = ["mkn2"]

    node = mantsNode(robots_list)

    rospy.spin()