#!/usr/bin/python3.8

from importlib_metadata import method_cache
import rospy
from mantsNode import mantsNode
from tansNode import tansNode


if __name__ == "__main__":
    rospy.init_node('runner')
    robots_list = rospy.get_param("~robots_list")
    print(robots_list)
    method = rospy.get_param("~method")
    robots_list = ["mkn2", "mkn5"]
    # robots_list = ["mkn2"]

    if method == "mants":
        node = mantsNode(robots_list)
    elif method == "tans":
        node = tansNode(robots_list)

    rospy.spin()