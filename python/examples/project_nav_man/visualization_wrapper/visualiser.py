#! /usr/bin/env python3

import rospy
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import String
import class_defs
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Point
import os

class Visualiser:
    def __init__(self):
        rospy.init_node('visualizer', log_level=rospy.DEBUG)
        rospy.logdebug("Visualizer node started")
        self.marker_pub = rospy.Publisher("/visualization_marker", Marker, queue_size = 10)
        self.marker_array_pub = rospy.Publisher("/visualization_marker_array", MarkerArray, queue_size=10)
        rospy.sleep(5)
        self.prev_pose=None
        self.clear_markers()
        # rospy.on_shutdown(self.clear_markers)
    
    def clear_markers(self):
        rospy.logdebug("Clearing markers")
        marker = Marker()
        marker.action = Marker.DELETEALL
        # self.marker_pub.publish(marker)
        array_marker = MarkerArray()
        array_marker.markers.append(marker)
        self.marker_array_pub.publish(array_marker)


    def make_name(self, data):
        return data.name + str(data.pose.position.x)[0] + str(data.pose.position.y)[0] + str(data.pose.position.z)[0] + str(data.pose.orientation.x)[0] + str(data.pose.orientation.y)[0] + str(data.pose.orientation.z)[0] + str(data.pose.orientation.w)[0]

    def make_line_name(self, pose1, pose2):
        return "line" + str(pose1.position.x)[0] + str(pose1.position.y)[0] + str(pose1.position.z)[0] + str(pose1.orientation.x)[0] + str(pose1.orientation.y)[0] + str(pose1.orientation.z)[0] + str(pose1.orientation.w)[0] + str(pose2.position.x)[0] + str(pose2.position.y)[0] + str(pose2.position.z)[0] + str(pose2.orientation.x)[0] + str(pose2.orientation.y)[0] + str(pose2.orientation.z)[0] + str(pose2.orientation.w)[0]
    
    def make_marker(self, data,marker_type,name=None):
        rospy.logdebug("Making marker")
        marker_shape={"node":3,"location":1,"object":2}
        marker_scale={"node":1,"location":0.5,"object":0.2}
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.type = marker_shape[marker_type]
        marker.action = 0
        marker.ns = self.make_name(data)
        
        marker.id = 1
        marker.scale.x = marker_scale[marker_type]
        marker.scale.y = marker_scale[marker_type]
        marker.scale.z = marker_scale[marker_type]
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        if marker_type == "node":
            marker.color.r = 1.0
        elif marker_type == "location":
            marker.color.g = 1.0
        elif marker_type == "object":
            marker.color.b = 1.0
        
        marker.color.a = 1.0
        marker.pose.position.x = data.pose.position.x
        marker.pose.position.y = data.pose.position.y
        marker.pose.position.z = data.pose.position.z
        marker.pose.orientation.x = data.pose.orientation.x
        marker.pose.orientation.y = data.pose.orientation.y
        marker.pose.orientation.z = data.pose.orientation.z
        marker.pose.orientation.w = data.pose.orientation.w
        marker.lifetime = rospy.Duration()

        text_marker = Marker()
        text_marker.header.frame_id = "map"
        text_marker.header.stamp = rospy.Time.now()
        text_marker.type = Marker.TEXT_VIEW_FACING
        text_marker.action = Marker.ADD
        text_marker.ns = marker.ns
        text_marker.id = 2
        text_marker.scale.z = 0.2  # Text size
        text_marker.color.r = 1.0
        text_marker.color.g = 1.0
        text_marker.color.b = 1.0
        text_marker.color.a = 1.0
        if marker_type == "node":
            offset = 0.75
        elif marker_type == "location":
            offset = 0.35
        else:
            offset = 0.25
        text_marker.pose.position.x = data.pose.position.x  # Same position as the main marker
        text_marker.pose.position.y = data.pose.position.y
        text_marker.pose.position.z = data.pose.position.z + offset  # Adjust z position for visibility
        text_marker.text = name  # Replace with the text you want to display
        text_marker.lifetime = rospy.Duration()
        marker_array = MarkerArray()
        marker_array.markers.append(marker)
        marker_array.markers.append(text_marker)
        return marker_array
    
    def remove_marker(self, data,pose2=None):
        rospy.logdebug("Removing marker")
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.type = 3
        marker.action = 2
        marker.id = 1
        marker.ns = self.make_name(data)
        if pose2:
            self.remove_line(data.pose,pose2)
        self.marker_pub.publish(marker)

    def remove_line(self, pose1, pose2):
        rospy.logdebug("Removing line")
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.type = Marker.LINE_LIST
        marker.action = 2
        marker.id = 1
        marker.ns = self.make_line_name(pose1,pose2)
        self.marker_pub.publish(marker)

    def publish_marker(self, marker): 
        rospy.logdebug("Publishing marker")
        if isinstance(marker, MarkerArray):
            self.marker_array_pub.publish(marker)
        else:
            self.marker_pub.publish(marker)
        
        


    def make_line(self,pose1,pose2,flag=0):
        rospy.logdebug("Making line")
        line_marker = Marker()
        line_marker.header.frame_id = "map"
        line_marker.header.stamp = rospy.Time.now()
        line_marker.ns = self.make_line_name(pose1,pose2)
        
        line_marker.id = 1
        line_marker.type = Marker.LINE_LIST
        line_marker.action = Marker.ADD
        line_marker.pose.orientation.w = 1.0
        line_marker.scale.x = 0.01  
        line_marker.color.a = 1.0
        line_marker.color.r = 1.0
        line_marker.color.g = 1.0
        line_marker.color.b = 1.0
        if flag==1:
            line_marker.scale.x = 0.05
            line_marker.color.r = 0.2
            line_marker.color.g = 0.5
            line_marker.color.b = 0.75
        line_marker.points.append(Point(x=pose1.position.x, y=pose1.position.y, z=pose1.position.z))  # Start point (marker1)
        line_marker.points.append(Point(x=pose2.position.x, y=pose2.position.y, z=pose2.position.z))  # End point (marker2)
        line_marker.lifetime = rospy.Duration()
        return line_marker


    def visualise_node(self, node):
        rospy.logdebug("Visualising node")
        marker = self.make_marker(node,"node",node.name)
        self.publish_marker(marker)
        if self.prev_pose!=None:
            self.publish_marker(self.make_line(self.prev_pose,node.pose,1))
        self.prev_pose = node.pose
        
        for location in node.locations:
            self.visualise_location(location, node.pose)

    
    def visualise_location(self, location, node_pose):
        rospy.logdebug("Visualising location")
        marker = self.make_marker(location,"location",location.name)
        self.publish_marker(marker)
        self.publish_marker(self.make_line(node_pose,location.pose))
        for obj in location.objects:
            self.visualise_object(obj, location.pose)
        
            
        
    def visualise_object(self, obj, location_pose):
        rospy.logdebug("Visualising object")
        marker = self.make_marker(obj,"object",obj.name)
        self.publish_marker(marker)
        self.publish_marker(self.make_line(location_pose,obj.pose))

    
        
