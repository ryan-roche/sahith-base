# Import pose from ROS
from geometry_msgs.msg import Pose

class Object:
    # Contains object name and pose of object
    def __init__(self, name, pose,location=None):
        self.name = name
        self.pose = pose
        self.location =location
    
    def add_location(self,location):
        self.location=location
        
    def get_pose(self):
        return self.pose

    def get_location(self):
        return self.location

class Semantic_location():
    # Contains object name and pose of semantic location
    def __init__(self, name, pose,node_name):
        self.name = name
        self.pose = pose
        self.objects = []
        self.node_name = node_name
    
    def add_object(self, object):
        self.objects.append(object)
    
    def get_objects(self):
        return self.objects

    def delete_object(self, object):
        self.objects.remove(object)
    
    def get_node_name(self):
        return self.node_name
    
    def get_pose(self):
        return self.pose
    
class Nodes():
    def __init__(self, name, pose):
        self.name = name
        self.pose = pose
        self.locations= []

    def add_location(self, location):
        self.locations.append(location)
    
    def get_locations(self):
        return self.locations

    def delete_location(self, location):
        self.locations.remove(location)
    
    def get_pose(self):
        return self.pose


