# Import pose from ROS
from geometry_msgs.msg import Pose

class Object:
    # Contains object name and pose of object
    def __init__(self, name, pose,bbox=None,location=None):
        self.name = name
        self.pose = pose
        self.location =location
        self.bbox=bbox
    
    def add_location(self,location):
        self.location=location
        
    def get_pose(self):
        return self.pose

    def get_location(self):
        return self.location
        
    def get_bbox(self):
        return self.bbox
        
class Semantic_location():
    # Contains object name and pose of semantic location
    def __init__(self, name, pose,node_name,bbox=None):
        self.name = name
        self.pose = pose
        self.objects = []
        self.node_name = node_name
        self.bbox=bbox
    
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
        
    def get_bbox(self):
        return self.bbox
    
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


