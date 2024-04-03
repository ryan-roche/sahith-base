#!/usr/bin/env python3

import rospy
from class_defs import Nodes, Semantic_location, Object
from geometry_msgs.msg import Pose, Point, Quaternion
from visualiser import Visualiser  # Import the Visualiser class

if __name__ == '__main__':
    

    visualiser = Visualiser()

    # Test Case 1 (1 semantic location with 2 objects)
    node1_pose = Pose(Point(0, 0, 0), Quaternion(0, 0, 0, 1))
    node1 = Nodes("Node1", node1_pose)

    loc1_pose = Pose(Point(1, 0, 0), Quaternion(0, 0, 0, 1))
    loc1 = Semantic_location("Location1", loc1_pose, "Node1")

    obj1_pose = Pose(Point(1, 0, 1), Quaternion(0, 0, 0, 1))
    obj1 = Object("Object1", obj1_pose, "Location1")

    obj2_pose = Pose(Point(1, 1, 1), Quaternion(0, 0, 0, 1))
    obj2 = Object("Object2", obj2_pose, "Location1")

    node1.add_location(loc1)
    loc1.add_object(obj1)
    loc1.add_object(obj2)

    visualiser.visualise_node(node1)
    input("Press Enter to add node2...")
    # Test Case 2 (1 semantic location with 3 objects)
    node2_pose = Pose(Point(5, 0, 0), Quaternion(0, 0, 0, 1))
    node2 = Nodes("Node2", node2_pose)

    loc2_pose = Pose(Point(4, 0, 0), Quaternion(0, 0, 0, 1))
    loc2 = Semantic_location("Location2", loc2_pose, "Node2")

    obj2_1_pose = Pose(Point(4, 0, 1), Quaternion(0, 0, 0, 1))
    obj2_1 = Object("Object2_1", obj2_1_pose, "Location2")

    obj2_2_pose = Pose(Point(4, 1, 1), Quaternion(0, 0, 0, 1))
    obj2_2 = Object("Object2_2", obj2_2_pose, "Location2")

    obj2_3_pose = Pose(Point(4, -1, 1), Quaternion(0, 0, 0, 1))
    obj2_3 = Object("Object2_3", obj2_3_pose, "Location2")

    node2.add_location(loc2)
    loc2.add_object(obj2_1)
    loc2.add_object(obj2_2)
    loc2.add_object(obj2_3)

    visualiser.visualise_node(node2)
    input("Press Enter to add node3...")
    # Test Case 3 (1 semantic location with 2 objects)
    node3_pose = Pose(Point(0, 5, 0), Quaternion(0, 0, 0, 1))
    node3 = Nodes("Node3", node3_pose)

    loc3_pose = Pose(Point(-1, 5, 0), Quaternion(0, 0, 0, 1))
    loc3 = Semantic_location("Location3", loc3_pose, "Node3")

    obj3_1_pose = Pose(Point(-1, 5, 1), Quaternion(0, 0, 0, 1))
    obj3_1 = Object("Object3_1", obj3_1_pose, "Location3")

    obj3_2_pose = Pose(Point(-1, 6, 1), Quaternion(0, 0, 0, 1))
    obj3_2 = Object("Object3_2", obj3_2_pose, "Location3")

    node3.add_location(loc3)
    loc3.add_object(obj3_1)
    loc3.add_object(obj3_2)

    visualiser.visualise_node(node3)
    input("Press Enter to add node4...")
    # Test Case 4 (1 semantic location with 3 objects)
    node4_pose = Pose(Point(0, 0, 5), Quaternion(0, 0, 0, 1))
    node4 = Nodes("Node4", node4_pose)

    loc4_pose = Pose(Point(0, 1, 5), Quaternion(0, 0, 0, 1))
    loc4 = Semantic_location("Location4", loc4_pose, "Node4")

    obj4_1_pose = Pose(Point(0, 1, 6), Quaternion(0, 0, 0, 1))
    obj4_1 = Object("Object4_1", obj4_1_pose, "Location4")

    obj4_2_pose = Pose(Point(0, 1, 7), Quaternion(0, 0, 0, 1))
    obj4_2 = Object("Object4_2", obj4_2_pose, "Location4")

    obj4_3_pose = Pose(Point(0, 1, 8), Quaternion(0, 0, 0, 1))
    obj4_3 = Object("Object4_3", obj4_3_pose, "Location4")

    node4.add_location(loc4)
    loc4.add_object(obj4_1)
    loc4.add_object(obj4_2)
    loc4.add_object(obj4_3)

    visualiser.visualise_node(node4)