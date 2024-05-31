# Copyright (c) 2022 Boston Dynamics, Inc.  All rights reserved.
#
# Downloading, reproducing, distributing or otherwise using the SDK Software
# is subject to the terms and conditions of the Boston Dynamics Software
# Development Kit License (20191101-BDSDK-SL).

"""Command line interface for graph nav with options to download/upload a map and to navigate a map. """

import argparse
import logging
import math
import os
import sys
import time
import cv2
import numpy as np
import supervision as sv

import google.protobuf.timestamp_pb2
import graph_nav_util
import grpc
import pickle

import bosdyn.client.channel
import bosdyn.client.util
from geometry_msgs.msg import Pose
from bosdyn.api import geometry_pb2, power_pb2, estop_pb2, robot_state_pb2, image_pb2, manipulation_api_pb2
from bosdyn.api.graph_nav import graph_nav_pb2, map_pb2, nav_pb2
from bosdyn.client.estop import EstopClient
from bosdyn.client.exceptions import ResponseError
from bosdyn.client.frame_helpers import get_a_tform_b,VISION_FRAME_NAME, get_odom_tform_body,GRAV_ALIGNED_BODY_FRAME_NAME,ODOM_FRAME_NAME
from bosdyn.client.graph_nav import GraphNavClient
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive, ResourceAlreadyClaimedError
from bosdyn.client.math_helpers import Quat, SE3Pose
from bosdyn.client.power import PowerClient, power_on, safe_power_off
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient,block_until_arm_arrives
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.image import ImageClient,pixel_to_camera_space
from bosdyn.client.manipulation_api_client import ManipulationApiClient

from grounded_sam_spot_package.grounded_sam_process import segment_image
from visualization_wrapper.visualiser import Visualiser  # Import the Visualiser class
from visualization_wrapper.class_defs import Nodes, Semantic_location, Object

g_image_click = None
g_image_display = None

def verify_estop(robot):
    """Verify the robot is not estopped"""

    client = robot.ensure_client(EstopClient.default_service_name)
    if client.get_status().stop_level != estop_pb2.ESTOP_LEVEL_NONE:
        error_message = "Robot is estopped. Please use an external E-Stop client, such as the" \
        " estop SDK example, to configure E-Stop."
        robot.logger.error(error_message)
        raise Exception(error_message)
    
def cv_mouse_callback(event, x, y, flags, param):
    global g_image_click, g_image_display
    clone = g_image_display.copy()
    if event == cv2.EVENT_LBUTTONUP:
        g_image_click = (x, y)
    else:
        # Draw some lines on the image.
        #print('mouse', x, y)
        color = (30, 30, 30)
        thickness = 2
        image_title = 'Click to grasp'
        height = clone.shape[0]
        width = clone.shape[1]
        cv2.line(clone, (0, y), (width, y), color, thickness)
        cv2.line(clone, (x, 0), (x, height), color, thickness)
        cv2.imshow(image_title, clone)

class GraphNavInterface(object):
    """GraphNav service command line interface."""

    def __init__(self, robot, upload_path):
        self._robot = robot

        # Force trigger timesync.
        self._robot.time_sync.wait_for_sync()

        # Create robot state and command clients.
        self._robot_command_client = self._robot.ensure_client(RobotCommandClient.default_service_name)
        self._robot_state_client = self._robot.ensure_client(RobotStateClient.default_service_name)

        # Create the client for the Graph Nav main service.
        self._graph_nav_client = self._robot.ensure_client(GraphNavClient.default_service_name)

        # Create a power client for the robot.
        self._power_client = self._robot.ensure_client(PowerClient.default_service_name)

        # Boolean indicating the robot's power state.
        power_state = self._robot_state_client.get_robot_state().power_state
        self._started_powered_on = (power_state.motor_power_state == power_state.STATE_ON)
        self._powered_on = self._started_powered_on

        # Number of attempts to wait before trying to re-power on.
        self._max_attempts_to_wait = 50

        # Store the most recent knowledge of the state of the robot based on rpc calls.
        self._current_graph = None
        self._current_edges = dict()  #maps to_waypoint to list(from_waypoint)
        self._current_waypoint_snapshots = dict()  # maps id to waypoint snapshot
        self._current_edge_snapshots = dict()  # maps id to edge snapshot
        self._current_annotation_name_to_wp_id = dict()

        # Filepath for uploading a saved graph's and snapshots too.
        if upload_path[-1] == "/":
            self._upload_filepath = upload_path[:-1]
        else:
            self._upload_filepath = upload_path

        self._command_dictionary = {
            '1': self._get_localization_state,
            '2': self._set_initial_localization_fiducial,
            '3': self._set_initial_localization_waypoint,
            '4': self._list_graph_waypoint_and_edge_ids,
            '5': self._upload_graph_and_snapshots,
            '6': self._navigate_to,
            '7': self._navigate_route,
            '8': self._navigate_to_anchor,
            '9': self._clear_graph,
            'p': self._pick_object_at_waypt,
            'd': self._drop_object_at_waypt,
            's': self._search_in_graph,
            'n': self._navigate_waypoints,
            'sv': self._save_modified_pkl_file
        }

        self.object_classes=["Bottle", "Clamp","Rubicks Cube","Brush","Umbrella","Mouse"]
        self.location_classes=["Floor","Table","Shelf"]
        self.thing_classes=self.object_classes+self.location_classes

        self._image_source=['hand_color_image']
        self.waypoint_nodes=[]
        self.visualizer=Visualiser()

    def _get_localization_state(self, *args):
        """Get the current localization and state of the robot."""
        state = self._graph_nav_client.get_localization_state()
        print('Got localization: \n%s' % str(state.localization))
        odom_tform_body = get_odom_tform_body(state.robot_kinematics.transforms_snapshot)
        print('Got robot state in kinematic odometry frame: \n%s' % str(odom_tform_body))

    def _set_initial_localization_fiducial(self, *args):
        """Trigger localization when near a fiducial."""
        robot_state = self._robot_state_client.get_robot_state()
        current_odom_tform_body = get_odom_tform_body(
            robot_state.kinematic_state.transforms_snapshot).to_proto()
        # Create an empty instance for initial localization since we are asking it to localize
        # based on the nearest fiducial.
        localization = nav_pb2.Localization()
        self._graph_nav_client.set_localization(initial_guess_localization=localization,
                                                ko_tform_body=current_odom_tform_body)

    def _set_initial_localization_waypoint(self, *args):
        """Trigger localization to a waypoint."""
        # Take the first argument as the localization waypoint.
        if len(args) < 1:
            # If no waypoint id is given as input, then return without initializing.
            print("No waypoint specified to initialize to.")
            return
        destination_waypoint = graph_nav_util.find_unique_waypoint_id(
            args[0][0], self._current_graph, self._current_annotation_name_to_wp_id)
        if not destination_waypoint:
            # Failed to find the unique waypoint id.
            return

        robot_state = self._robot_state_client.get_robot_state()
        current_odom_tform_body = get_odom_tform_body(
            robot_state.kinematic_state.transforms_snapshot).to_proto()
        # Create an initial localization to the specified waypoint as the identity.
        localization = nav_pb2.Localization()
        localization.waypoint_id = destination_waypoint
        localization.waypoint_tform_body.rotation.w = 1.0
        self._graph_nav_client.set_localization(
            initial_guess_localization=localization,
            # It's hard to get the pose perfect, search +/-20 deg and +/-20cm (0.2m).
            max_distance=0.2,
            max_yaw=20.0 * math.pi / 180.0,
            fiducial_init=graph_nav_pb2.SetLocalizationRequest.FIDUCIAL_INIT_NO_FIDUCIAL,
            ko_tform_body=current_odom_tform_body)

    def _list_graph_waypoint_and_edge_ids(self, *args):
        """List the waypoint ids and edge ids of the graph currently on the robot."""

        # Download current graph
        graph = self._graph_nav_client.download_graph()
        if graph is None:
            print("Empty graph.")
            return
        self._current_graph = graph

        localization_id = self._graph_nav_client.get_localization_state().localization.waypoint_id

        # Update and print waypoints and edges
        self._current_annotation_name_to_wp_id, self._current_edges = graph_nav_util.update_waypoints_and_edges(
            graph, localization_id)

    def _upload_graph_and_snapshots(self, *args):
        """Upload the graph and snapshots to the robot."""
        print("Loading the graph from disk into local storage...")
        with open(self._upload_filepath + "/graph", "rb") as graph_file:
            # Load the graph from disk.
            data = graph_file.read()
            self._current_graph = map_pb2.Graph()
            self._current_graph.ParseFromString(data)
            print("Loaded graph has {} waypoints and {} edges".format(
                len(self._current_graph.waypoints), len(self._current_graph.edges)))
        for waypoint in self._current_graph.waypoints:
            # Load the waypoint snapshots from disk.
            with open(self._upload_filepath + "/waypoint_snapshots/{}".format(waypoint.snapshot_id),
                      "rb") as snapshot_file:
                waypoint_snapshot = map_pb2.WaypointSnapshot()
                waypoint_snapshot.ParseFromString(snapshot_file.read())
                self._current_waypoint_snapshots[waypoint_snapshot.id] = waypoint_snapshot
        for edge in self._current_graph.edges:
            if len(edge.snapshot_id) == 0:
                continue
            # Load the edge snapshots from disk.
            with open(self._upload_filepath + "/edge_snapshots/{}".format(edge.snapshot_id),
                      "rb") as snapshot_file:
                edge_snapshot = map_pb2.EdgeSnapshot()
                edge_snapshot.ParseFromString(snapshot_file.read())
                self._current_edge_snapshots[edge_snapshot.id] = edge_snapshot
        # Upload the graph to the robot.
        print("Uploading the graph and snapshots to the robot...")
        true_if_empty = not len(self._current_graph.anchoring.anchors)
        response = self._graph_nav_client.upload_graph(graph=self._current_graph,
                                                       generate_new_anchoring=true_if_empty)
        # Upload the snapshots to the robot.
        for snapshot_id in response.unknown_waypoint_snapshot_ids:
            waypoint_snapshot = self._current_waypoint_snapshots[snapshot_id]
            self._graph_nav_client.upload_waypoint_snapshot(waypoint_snapshot)
            print("Uploaded {}".format(waypoint_snapshot.id))
        for snapshot_id in response.unknown_edge_snapshot_ids:
            edge_snapshot = self._current_edge_snapshots[snapshot_id]
            self._graph_nav_client.upload_edge_snapshot(edge_snapshot)
            print("Uploaded {}".format(edge_snapshot.id))

        # The upload is complete! Check that the robot is localized to the graph,
        # and if it is not, prompt the user to localize the robot before attempting
        # any navigation commands.
        localization_state = self._graph_nav_client.get_localization_state()
        if not localization_state.localization.waypoint_id:
            # The robot is not localized to the newly uploaded graph.
            print("\n")
            print("Upload complete! The robot is currently not localized to the map; please localize", \
                   "the robot using commands (2) or (3) before attempting a navigation command.")
        print("reading filepath")
        with open(self._upload_filepath + '/semantic_locations.pkl', 'rb') as file:  # Open file in read-binary mode
            self.waypoint_nodes = pickle.load(file)
        
        for waypoint_node in self.waypoint_nodes:
            self.visualizer.visualise_node(waypoint_node)

    def _save_modified_pkl_file(self,*args):
        with open(self._upload_filepath + '/semantic_locations.pkl', 'wb') as file:
            pickle.dump(self.waypoint_nodes, file)

    def _navigate_to_anchor(self, *args):
        """Navigate to a pose in seed frame, using anchors."""
        # The following options are accepted for arguments: [x, y], [x, y, yaw], [x, y, z, yaw],
        # [x, y, z, qw, qx, qy, qz].
        # When a value for z is not specified, we use the current z height.
        # When only yaw is specified, the quaternion is constructed from the yaw.
        # When yaw is not specified, an identity quaternion is used.

        if len(args) < 1 or len(args[0]) not in [2, 3, 4, 7]:
            print("Invalid arguments supplied.")
            return

        seed_T_goal = SE3Pose(float(args[0][0]), float(args[0][1]), 0.0, Quat())

        if len(args[0]) in [4, 7]:
            seed_T_goal.z = float(args[0][2])
        else:
            localization_state = self._graph_nav_client.get_localization_state()
            if not localization_state.localization.waypoint_id:
                print("Robot not localized")
                return
            seed_T_goal.z = localization_state.localization.seed_tform_body.position.z

        if len(args[0]) == 3:
            seed_T_goal.rot = Quat.from_yaw(float(args[0][2]))
        elif len(args[0]) == 4:
            seed_T_goal.rot = Quat.from_yaw(float(args[0][3]))
        elif len(args[0]) == 7:
            seed_T_goal.rot = Quat(w=float(args[0][3]), x=float(args[0][4]), y=float(args[0][5]),
                                   z=float(args[0][6]))

        if not self.toggle_power(should_power_on=True):
            print("Failed to power on the robot, and cannot complete navigate to request.")
            return

        nav_to_cmd_id = None
        # Navigate to the destination.
        is_finished = False
        while not is_finished:
            # Issue the navigation command about twice a second such that it is easy to terminate the
            # navigation command (with estop or killing the program).
            try:
                nav_to_cmd_id = self._graph_nav_client.navigate_to_anchor(
                    seed_T_goal.to_proto(), 1.0, command_id=nav_to_cmd_id)
            except ResponseError as e:
                print("Error while navigating {}".format(e))
                break
            time.sleep(.5)  # Sleep for half a second to allow for command execution.
            # Poll the robot for feedback to determine if the navigation command is complete. Then sit
            # the robot down once it is finished.
            is_finished = self._check_success(nav_to_cmd_id)

        # Power off the robot if appropriate.
        if self._powered_on and not self._started_powered_on:
            # Sit the robot down + power off after the navigation command is complete.
            self.toggle_power(should_power_on=False)

    def _navigate_to(self, *args):
        """Navigate to a specific waypoint."""
        # Take the first argument as the destination waypoint.
        if len(args) < 1:
            # If no waypoint id is given as input, then return without requesting navigation.
            print("No waypoint provided as a destination for navigate to.")
            return

        destination_waypoint = graph_nav_util.find_unique_waypoint_id(
            args[0][0], self._current_graph, self._current_annotation_name_to_wp_id)
        if not destination_waypoint:
            # Failed to find the appropriate unique waypoint id for the navigation command.
            return
        if not self.toggle_power(should_power_on=True):
            print("Failed to power on the robot, and cannot complete navigate to request.")
            return

        nav_to_cmd_id = None
        # Navigate to the destination waypoint.
        is_finished = False
        while not is_finished:
            # Issue the navigation command about twice a second such that it is easy to terminate the
            # navigation command (with estop or killing the program).
            try:
                nav_to_cmd_id = self._graph_nav_client.navigate_to(destination_waypoint, 1.0,
                                                                   command_id=nav_to_cmd_id)
            except ResponseError as e:
                print("Error while navigating {}".format(e))
                break
            time.sleep(.5)  # Sleep for half a second to allow for command execution.
            # Poll the robot for feedback to determine if the navigation command is complete. Then sit
            # the robot down once it is finished.
            is_finished = self._check_success(nav_to_cmd_id)

        # Power off the robot if appropriate.
        if self._powered_on and not self._started_powered_on:
            # Sit the robot down + power off after the navigation command is complete.
            self.toggle_power(should_power_on=False)

    def _navigate_route(self, *args):
        """Navigate through a specific route of waypoints."""
        if len(args) < 1 or len(args[0]) < 1:
            # If no waypoint ids are given as input, then return without requesting navigation.
            print("No waypoints provided for navigate route.")
            return
        waypoint_ids = args[0]
        for i in range(len(waypoint_ids)):
            waypoint_ids[i] = graph_nav_util.find_unique_waypoint_id(
                waypoint_ids[i], self._current_graph, self._current_annotation_name_to_wp_id)
            if not waypoint_ids[i]:
                # Failed to find the unique waypoint id.
                return

        edge_ids_list = []
        all_edges_found = True
        # Attempt to find edges in the current graph that match the ordered waypoint pairs.
        # These are necessary to create a valid route.
        for i in range(len(waypoint_ids) - 1):
            start_wp = waypoint_ids[i]
            end_wp = waypoint_ids[i + 1]
            edge_id = self._match_edge(self._current_edges, start_wp, end_wp)
            if edge_id is not None:
                edge_ids_list.append(edge_id)
            else:
                all_edges_found = False
                print("Failed to find an edge between waypoints: ", start_wp, " and ", end_wp)
                print(
                    "List the graph's waypoints and edges to ensure pairs of waypoints has an edge."
                )
                break

        if all_edges_found:
            if not self.toggle_power(should_power_on=True):
                print("Failed to power on the robot, and cannot complete navigate route request.")
                return

            # Navigate a specific route.
            route = self._graph_nav_client.build_route(waypoint_ids, edge_ids_list)
            is_finished = False
            while not is_finished:
                # Issue the route command about twice a second such that it is easy to terminate the
                # navigation command (with estop or killing the program).
                nav_route_command_id = self._graph_nav_client.navigate_route(
                    route, cmd_duration=1.0)
                time.sleep(.5)  # Sleep for half a second to allow for command execution.
                # Poll the robot for feedback to determine if the route is complete. Then sit
                # the robot down once it is finished.
                is_finished = self._check_success(nav_route_command_id)

            # Power off the robot if appropriate.
            if self._powered_on and not self._started_powered_on:
                # Sit the robot down + power off after the navigation command is complete.
                self.toggle_power(should_power_on=False)

    def inter_over_area(self, obj, semantic_location):
        # Get bounding boxes
        obj_bbox = obj.get_bbox()  # (x1, y1, x2, y2)
        location_bbox = semantic_location.get_bbox()  # (x1, y1, x2, y2)
        
        # Calculate the intersection box coordinates
        ix1 = max(obj_bbox[0], location_bbox[0])
        iy1 = max(obj_bbox[1], location_bbox[1])
        ix2 = min(obj_bbox[2], location_bbox[2])
        iy2 = min(obj_bbox[3], location_bbox[3])
        
        # Check if there is an intersection
        if ix1 < ix2 and iy1 < iy2:
            intersection_area = (ix2 - ix1) * (iy2 - iy1)
        else:
            intersection_area = 0
        
        # Calculate each box's area
        obj_area = (obj_bbox[2] - obj_bbox[0]) * (obj_bbox[3] - obj_bbox[1])
        location_area = (location_bbox[2] - location_bbox[0]) * (location_bbox[3] - location_bbox[1])
        
        # Calculate Intersection over Union
        if obj_area == 0:
            return 0  # to avoid division by zero if both areas are zero
        else:
            iou = intersection_area / obj_area
            return iou   

    def get_dist(self,obj,semantic_location):
        obj_pose=obj.get_pose()
        loc_pose=semantic_location.get_pose()
        dist= np.sqrt(np.square(obj_pose.position.x-loc_pose.position.x)+np.square(obj_pose.position.y-loc_pose.position.y)+np.square(obj_pose.position.z-loc_pose.position.z))
        return dist

    def navigate_waypoint_power_on(self,waypoint_id):
        """Navigate to a specific waypoint."""
        destination_waypoint = graph_nav_util.find_unique_waypoint_id(
            waypoint_id, self._current_graph, self._current_annotation_name_to_wp_id)
        if not destination_waypoint:
            # Failed to find the appropriate unique waypoint id for the navigation command.
            return
        if not self.toggle_power(should_power_on=True):
            print("Failed to power on the robot, and cannot complete navigate to request.")
            return

        nav_to_cmd_id = None
        # Navigate to the destination waypoint.
        is_finished = False
        while not is_finished:
            # Issue the navigation command about twice a second such that it is easy to terminate the
            # navigation command (with estop or killing the program).
            try:
                nav_to_cmd_id = self._graph_nav_client.navigate_to(destination_waypoint, 1.0,
                                                                   command_id=nav_to_cmd_id)
            except ResponseError as e:
                print("Error while navigating {}".format(e))
                break
            time.sleep(.5)  # Sleep for half a second to allow for command execution.
            # Poll the robot for feedback to determine if the navigation command is complete. Then sit
            # the robot down once it is finished.
            is_finished = self._check_success(nav_to_cmd_id)


    def _navigate_waypoints(self,*args):
        for i,waypoint in enumerate(self.waypoint_nodes):
            print("Going to: "+ waypoint.name)
            waypoint_id =self._current_annotation_name_to_wp_id[waypoint.name]
            if not waypoint_id:
                continue
            self.navigate_waypoint_power_on( waypoint_id)
            self._capt_objects_waypt(i)
    
    def _capt_objects_waypt(self,node_num):
        # open gripper
        waypoint=self.waypoint_nodes[node_num]
        gripper_command = RobotCommandBuilder.claw_gripper_open_fraction_command(
            1.0)
        command = RobotCommandBuilder.build_synchro_command(gripper_command)
        cmd_id = self._robot_command_client.robot_command(command)

        # Wait for gripper to open
        time.sleep(1.5)
        image_client = self._robot.ensure_client(ImageClient.default_service_name)
        image_requests=[]
        image_source=['hand_depth_in_hand_color_frame','hand_color_image']
        for i,source in enumerate(image_source):
            image_requests.append(image_pb2.ImageRequest(image_source_name=image_source[i],quality_percent=100))

        image_responses = image_client.get_image(image_requests)
        if len(image_responses) < 2:
            print('Error: failed to get images.')
            return False

        # Depth is a raw bytestream
        cv_depth = np.frombuffer(image_responses[0].shot.image.data, dtype=np.uint16)
        cv_depth = cv_depth.reshape(image_responses[0].shot.image.rows,
                                    image_responses[0].shot.image.cols)

        # Visual is a JPEG
        cv_visual = cv2.imdecode(np.frombuffer(image_responses[1].shot.image.data, dtype=np.uint8), -1)

        # Convert the visual image from a single channel to RGB so we can add color
        visual_rgb = cv_visual if len(cv_visual.shape) == 3 else cv2.cvtColor(
            cv_visual, cv2.COLOR_GRAY2RGB)

        detections,annotated_image=segment_image(visual_rgb,self.object_classes)
        obj_list=[]
        for i in range(len(detections)):
            depth_seg=detections.mask[i]*cv_depth
            mean_depth=np.mean(depth_seg[depth_seg!=0])
        
            object_name = self.thing_classes[detections.class_id[i]]  # Extract object name

            try:
                min_val_depth=np.min(depth_seg[depth_seg!=0])
            except ValueError:
                min_val_depth=0
                object_name = self.thing_classes[detections[i].class_id[0]]  # Extract object name

            non_zero_indices = np.argwhere(depth_seg != 0)
            # Calculate the mean of these indices
            
            if non_zero_indices.size==0:
                print(object_name+ " out of bound of depth camera's field of view")
                continue
            try:
                mean_location = non_zero_indices.mean(axis=0)
            except RuntimeWarning:
                print(object_name+ " out of bound of depth camera's field of view")
                continue

            mean_location=mean_location.astype(np.int32)
            
            print(f'Found object "{object_name}" at image location ({mean_location[1]}, {mean_location[0]})')

            pick_vec = geometry_pb2.Vec2(x=mean_location[1], y=mean_location[0])

            # Build the proto for each point
            grasp = manipulation_api_pb2.PickObjectInImage(
                pixel_xy=pick_vec, 
                transforms_snapshot_for_camera=image_responses[1].shot.transforms_snapshot,
                frame_name_image_sensor=image_responses[1].shot.frame_name_image_sensor,
                camera_model=image_responses[1].source.pinhole)
            
            depth_point=cv_depth[mean_location[1],mean_location[0]]

            
            tform_snapshot = image_responses[1].shot.transforms_snapshot
            if(min_val_depth<2000): #depth less than 1 meter
                pixel_point=pixel_to_camera_space(image_responses[1],mean_location[1],mean_location[0],depth_point/1000)
                cam_to_world_tform = get_a_tform_b(tform_snapshot,ODOM_FRAME_NAME,image_responses[1].shot.frame_name_image_sensor)
                world_coord=cam_to_world_tform.transform_cloud(pixel_point)
                
                print("added "+str(object_name)+" at "+str(world_coord) +" to the graph")

                annotated_temp=cv2.circle(annotated_image,(mean_location[1], mean_location[0]),radius=20,color=(0,0,255),thickness=-1)
                cv2.imshow(str(object_name)+" segmentation",annotated_temp)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

                change=input("Should we change "+object_name+": ")
                if change=="":
                    print("Object name= "+object_name)
                elif change=="skip":
                    print("Object "+object_name+" skipped")
                    continue
                else:
                    print("Object "+object_name+" changed to "+ change)
                    object_name=change
                temp_pose=Pose()
                temp_pose.position.x,temp_pose.position.y,temp_pose.position.z=world_coord
                temp_pose.orientation.x=1

                if object_name in self.object_classes:
                    obj_node= Object(object_name,temp_pose,detections.xyxy[i])
                    obj_list.append(obj_node)
                annotated_image=cv2.circle(annotated_image,(mean_location[1], mean_location[0]),radius=20,color=(0,0,255),thickness=-1)
                
                # Upload the modified graph back to the robot
        for semantic_location in self.waypoint_nodes[node_num].get_locations():
            semantic_location.delete_all_objects()
        
        for obj in obj_list:
            max_inter,max_location,min_dist,closest_loc=0,None,1e7,None
            for semantic_location in self.waypoint_nodes[node_num].get_locations():
                inter=self.inter_over_area(obj,semantic_location)
                dist=self.get_dist(obj,semantic_location)
                if inter>max_inter:
                    max_location=semantic_location
                    max_inter=inter
                if dist<min_dist:
                    closest_loc=semantic_location
                    min_dist=dist
            if max_location:
                max_location.add_object(obj)
                obj.add_location(max_location)
            elif min_dist<1e7:
                closest_loc.add_object(obj)
                obj.add_location(closest_loc)

        #delete old waypoint visualization
        self.visualizer.clear_markers()
        for waypt in self.waypoint_nodes:
            self.visualizer.visualise_node(waypt)
        #self.waypoint_nodes.append(waypoint_node)

    def _drop_object_at_waypt(self,*args):
        '''Navigate to a waypoint and drop object'''
       # Take the first argument as the destination waypoint.
        if len(args) < 1:
            # If no waypoint id is given as input, then return without requesting navigation.
            print("No waypoint provided as a destination for navigate to.")
            return

        destination_waypoint = graph_nav_util.find_unique_waypoint_id(
            args[0][0], self._current_graph, self._current_annotation_name_to_wp_id)
        if not destination_waypoint:
            # Failed to find the appropriate unique waypoint id for the navigation command.
            return
        if not self.toggle_power(should_power_on=True):
            print("Failed to power on the robot, and cannot complete navigate to request.")
            return

        nav_to_cmd_id = None
        # Navigate to the destination waypoint.
        is_finished = False
        while not is_finished:
            # Issue the navigation command about twice a second such that it is easy to terminate the
            # navigation command (with estop or killing the program).
            try:
                nav_to_cmd_id = self._graph_nav_client.navigate_to(destination_waypoint, 1.0,
                                                                   command_id=nav_to_cmd_id)
            except ResponseError as e:
                print("Error while navigating {}".format(e))
                break
            time.sleep(.5)  # Sleep for half a second to allow for command execution.
            # Poll the robot for feedback to determine if the navigation command is complete. Then sit
            # the robot down once it is finished.
            is_finished = self._check_success(nav_to_cmd_id)
        self.drop_object(self,*args)
        # if self._powered_on and not self._started_powered_on:
        #     # Sit the robot down + power off after the navigation command is complete.
        #     self.toggle_power(should_power_on=False)   

    def drop_object(self,*args):
        print('Arrived at goal, dropping object...')
        # Do an arm-move to gently put the object down.
        # Build a position to move the arm to (in meters, relative to and expressed in the gravity aligned body frame).
        x = 0.75
        y = 0
        z = -0.25
        hand_ewrt_flat_body = geometry_pb2.Vec3(x=x, y=y, z=z)

        # Point the hand straight down with a quaternion.
        qw = 0.707
        qx = 0
        qy = 0.707
        qz = 0
        flat_body_Q_hand = geometry_pb2.Quaternion(w=qw, x=qx, y=qy, z=qz)

        flat_body_tform_hand = geometry_pb2.SE3Pose(
            position=hand_ewrt_flat_body, rotation=flat_body_Q_hand)

        robot_state = self._robot_state_client.get_robot_state()
        vision_tform_flat_body = get_a_tform_b(
            robot_state.kinematic_state.transforms_snapshot,
            VISION_FRAME_NAME,
            GRAV_ALIGNED_BODY_FRAME_NAME)

        vision_tform_hand_at_drop = vision_tform_flat_body * SE3Pose.from_obj(
            flat_body_tform_hand)
        
         # duration in seconds
        seconds = 1

        arm_command = RobotCommandBuilder.arm_pose_command(
            vision_tform_hand_at_drop.x, vision_tform_hand_at_drop.y,
            vision_tform_hand_at_drop.z, vision_tform_hand_at_drop.rot.w,
            vision_tform_hand_at_drop.rot.x, vision_tform_hand_at_drop.rot.y,
            vision_tform_hand_at_drop.rot.z, VISION_FRAME_NAME,
            seconds)

        # Keep the gripper closed.
        gripper_command = RobotCommandBuilder.claw_gripper_open_fraction_command(
            0.0)

        # Combine the arm and gripper commands into one RobotCommand
        command = RobotCommandBuilder.build_synchro_command(
            gripper_command, arm_command)

        # Send the request
        cmd_id = self._robot_command_client.robot_command(command)

        # Wait until the arm arrives at the goal.
        block_until_arm_arrives(self._robot_command_client, cmd_id)

        # Open the gripper
        gripper_command = RobotCommandBuilder.claw_gripper_open_fraction_command(
            1.0)
        command = RobotCommandBuilder.build_synchro_command(gripper_command)
        cmd_id = self._robot_command_client.robot_command(command)

        # Wait for the dogtoy to fall out
        time.sleep(1.5)

        # Stow the arm.
        stow_cmd = RobotCommandBuilder.arm_stow_command()
        self._robot_command_client.robot_command(stow_cmd)

        time.sleep(1)
        
    def _pick_object(self,*args):
        waypoint_name=self._search_in_graph(*args)
        destination_waypoint = graph_nav_util.find_unique_waypoint_id(
            waypoint_name, self._current_graph, self._current_annotation_name_to_wp_id)
        if not destination_waypoint:
            # Failed to find the appropriate unique waypoint id for the navigation command.
            return
        if not self.toggle_power(should_power_on=True):
            print("Failed to power on the robot, and cannot complete navigate to request.")
            return

        nav_to_cmd_id = None
        # Navigate to the destination waypoint.
        is_finished = False
        while not is_finished:
            # Issue the navigation command about twice a second such that it is easy to terminate the
            # navigation command (with estop or killing the program).
            try:
                nav_to_cmd_id = self._graph_nav_client.navigate_to(destination_waypoint, 1.0,
                                                                   command_id=nav_to_cmd_id)
            except ResponseError as e:
                print("Error while navigating {}".format(e))
                break
            time.sleep(.5)  # Sleep for half a second to allow for command execution.
            # Poll the robot for feedback to determine if the navigation command is complete. Then sit
            # the robot down once it is finished.
            is_finished = self._check_success(nav_to_cmd_id)

        
        self.arm_object_grasp(args[0][0])
        # Carry object
        self.carry_object(self,*args)



    def _search_in_graph(self,*args):
        object_name=None
        location_name=None
        if len(args) < 1:
            # If no waypoint id is given as input, then return without requesting navigation.
            print("No object or waypoint provided.")
            return
        if args[0][0] in self.object_classes:
            object_name=args[0][0]

        if args[0][0] in self.location_classes:
            location_name=args[0][0]

        if len(self.waypoint_nodes)==0:
            print("waypoint list is empty")
            return
        if not object_name and not location_name:
            print("Location nama and object name not specified")
            return
        waypoint_name=None
        if location_name:
            for node in self.waypoint_nodes:
                for location in node.get_locations():
                    if location.name==location_name:
                        waypoint_name=node.name
                        return waypoint_name
                        
        if object_name:
            for node in self.waypoint_nodes:
                for location in node.get_locations():
                    for object in location.get_objects():
                        if object.name==object_name:
                            waypoint_name==node.name
                            return waypoint_name

    def _pick_object_at_waypt(self,*args):
        '''Navigate to a waypoint and pick object'''
       # Take the first argument as the destination waypoint.
        if len(args) < 1:
            # If no waypoint id is given as input, then return without requesting navigation.
            print("No waypoint provided as a destination for navigate to.")
            return

        destination_waypoint = graph_nav_util.find_unique_waypoint_id(
            args[0][0], self._current_graph, self._current_annotation_name_to_wp_id)
        if not destination_waypoint:
            # Failed to find the appropriate unique waypoint id for the navigation command.
            return
        if not self.toggle_power(should_power_on=True):
            print("Failed to power on the robot, and cannot complete navigate to request.")
            return

        nav_to_cmd_id = None
        # Navigate to the destination waypoint.
        is_finished = False
        while not is_finished:
            # Issue the navigation command about twice a second such that it is easy to terminate the
            # navigation command (with estop or killing the program).
            try:
                nav_to_cmd_id = self._graph_nav_client.navigate_to(destination_waypoint, 1.0,
                                                                   command_id=nav_to_cmd_id)
            except ResponseError as e:
                print("Error while navigating {}".format(e))
                break
            time.sleep(.5)  # Sleep for half a second to allow for command execution.
            # Poll the robot for feedback to determine if the navigation command is complete. Then sit
            # the robot down once it is finished.
            is_finished = self._check_success(nav_to_cmd_id)

        object_name=input("what object are you interested in?")
        self.arm_object_grasp(object_name)
        # Carry object
        self.carry_object(self,*args)

        # if self._powered_on and not self._started_powered_on:
        #     # Sit the robot down + power off after the navigation command is complete.
        #     self.toggle_power(should_power_on=False)   

    def arm_object_grasp(self,obj_to_grasp=None):
        """A simple example of using the Boston Dynamics API to command Spot's arm."""

        image_client = self._robot.ensure_client(ImageClient.default_service_name)
        manipulation_api_client = self._robot.ensure_client(ManipulationApiClient.default_service_name)
        # open gripper
        gripper_command = RobotCommandBuilder.claw_gripper_open_fraction_command(
            1.0)
        command = RobotCommandBuilder.build_synchro_command(gripper_command)
        cmd_id = self._robot_command_client.robot_command(command)

        # Wait for gripper to open
        time.sleep(1.5)
        
        image_requests=[]
        for i,source in enumerate(self._image_source):
            image_requests.append(image_pb2.ImageRequest(image_source_name=self._image_source[i],quality_percent=100))
            self._robot.logger.info('Getting an image from: ' + self._image_source[i])
        image_responses = image_client.get_image(image_requests)

        if len(image_responses) != 1:
            print('Got invalid number of images: ' + str(len(image_responses)))
            print(image_responses)
            assert False

        cv_visual = cv2.imdecode(np.frombuffer(image_responses[0].shot.image.data, dtype=np.uint8), -1)

        # Convert the visual image from a single channel to RGB so we can add color
        visual_rgb = cv_visual if len(cv_visual.shape) == 3 else cv2.cvtColor(
            cv_visual, cv2.COLOR_GRAY2RGB)

        detections,annotated_image=segment_image(visual_rgb,self.thing_classes)
        obj_list=[]
        pick_vec=None
        for i in range(len(detections)):
            object_name = self.thing_classes[detections.class_id[i]]  # Extract object name
            if object_name==obj_to_grasp:
                mask_annotator = sv.MaskAnnotator()
                annotated_segment = mask_annotator.annotate(scene=annotated_image, detections=detections[i])

                depth_seg=detections.mask[i]
                non_zero_indices = np.argwhere(depth_seg != 0)
                # Calculate the mean of these indices
                
                if non_zero_indices.size==0:
                    print(object_name+ " out of bound of depth camera's field of view")
                    continue
                try:
                    mean_location = non_zero_indices.mean(axis=0)
                except RuntimeWarning:
                    print(object_name+ " out of bound of depth camera's field of view")
                    continue

                mean_location=mean_location.astype(np.int32)
                
                print(f'Found object "{object_name}" at image location ({mean_location[1]}, {mean_location[0]})')
                
                pick_vec = geometry_pb2.Vec2(x=mean_location[1], y=mean_location[0])
                annotated_segment=cv2.circle(annotated_segment,(mean_location[1], mean_location[0]),radius=20,color=(0,0,255),thickness=-1)
                cv2.imshow("segmentations",annotated_segment)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                break
        breakpoint()
        if not pick_vec:
            print("required object not found")
            return

        image = image_responses[0]
        # Build the proto
        grasp = manipulation_api_pb2.PickObjectInImage(
            pixel_xy=pick_vec, transforms_snapshot_for_camera=image.shot.transforms_snapshot,
            frame_name_image_sensor=image.shot.frame_name_image_sensor,
            camera_model=image.source.pinhole)


        grasp.grasp_params.grasp_params_frame_name = VISION_FRAME_NAME

        # Ask the robot to pick up the object
        grasp_request = manipulation_api_pb2.ManipulationApiRequest(pick_object_in_image=grasp)

        # Send the request
        cmd_response = manipulation_api_client.manipulation_api_command(
            manipulation_api_request=grasp_request)

        # Get feedback from the robot
        while True:
            feedback_request = manipulation_api_pb2.ManipulationApiFeedbackRequest(
                manipulation_cmd_id=cmd_response.manipulation_cmd_id)

            # Send the request
            response = manipulation_api_client.manipulation_api_feedback_command(
                manipulation_api_feedback_request=feedback_request)

            print('Current state: ',
                manipulation_api_pb2.ManipulationFeedbackState.Name(response.current_state))

            if response.current_state == manipulation_api_pb2.MANIP_STATE_GRASP_SUCCEEDED or response.current_state == manipulation_api_pb2.MANIP_STATE_GRASP_FAILED:
                break

            time.sleep(0.25)

        self._robot.logger.info('Finished grasp.')
        # time.sleep(4.0)

        # self._robot.logger.info('Sitting down and turning off.')

        # # Power the robot off. By specifying "cut_immediately=False", a safe power off command
        # # is issued to the robot. This will attempt to sit the robot before powering off.
        # self._robot.power_off(cut_immediately=False, timeout_sec=20)
        # assert not self._robot.is_powered_on(), "Robot power off failed."
        # self._robot.logger.info("Robot safely powered off.")

    def carry_object(self,*args):
                    # Move the arm to a carry position.
        print('')
        print('Grasp finished, moving arm to carry position...')
        carry_cmd = RobotCommandBuilder.arm_carry_command()
        self._robot_command_client.robot_command(carry_cmd)

        # Wait for the carry command to finish
        time.sleep(0.75)


       
    def _clear_graph(self, *args):
        """Clear the state of the map on the robot, removing all waypoints and edges."""
        return self._graph_nav_client.clear_graph()

    def toggle_power(self, should_power_on):
        """Power the robot on/off dependent on the current power state."""
        is_powered_on = self.check_is_powered_on()
        if not is_powered_on and should_power_on:
            # Power on the robot up before navigating when it is in a powered-off state.
            power_on(self._power_client)
            motors_on = False
            while not motors_on:
                future = self._robot_state_client.get_robot_state_async()
                state_response = future.result(
                    timeout=10)  # 10 second timeout for waiting for the state response.
                if state_response.power_state.motor_power_state == robot_state_pb2.PowerState.STATE_ON:
                    motors_on = True
                else:
                    # Motors are not yet fully powered on.
                    time.sleep(.25)
        elif is_powered_on and not should_power_on:
            # Safe power off (robot will sit then power down) when it is in a
            # powered-on state.
            safe_power_off(self._robot_command_client, self._robot_state_client)
        else:
            # Return the current power state without change.
            return is_powered_on
        # Update the locally stored power state.
        self.check_is_powered_on()
        return self._powered_on

    def check_is_powered_on(self):
        """Determine if the robot is powered on or off."""
        power_state = self._robot_state_client.get_robot_state().power_state
        self._powered_on = (power_state.motor_power_state == power_state.STATE_ON)
        return self._powered_on

    def _check_success(self, command_id=-1):
        """Use a navigation command id to get feedback from the robot and sit when command succeeds."""
        if command_id == -1:
            # No command, so we have no status to check.
            return False
        status = self._graph_nav_client.navigation_feedback(command_id)
        if status.status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_REACHED_GOAL:
            # Successfully completed the navigation commands!
            return True
        elif status.status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_LOST:
            print("Robot got lost when navigating the route, the robot will now sit down.")
            return True
        elif status.status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_STUCK:
            print("Robot got stuck when navigating the route, the robot will now sit down.")
            return True
        elif status.status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_ROBOT_IMPAIRED:
            print("Robot is impaired.")
            return True
        else:
            # Navigation command is not complete yet.
            return False

    def _match_edge(self, current_edges, waypoint1, waypoint2):
        """Find an edge in the graph that is between two waypoint ids."""
        # Return the correct edge id as soon as it's found.
        for edge_to_id in current_edges:
            for edge_from_id in current_edges[edge_to_id]:
                if (waypoint1 == edge_to_id) and (waypoint2 == edge_from_id):
                    # This edge matches the pair of waypoints! Add it the edge list and continue.
                    return map_pb2.Edge.Id(from_waypoint=waypoint2, to_waypoint=waypoint1)
                elif (waypoint2 == edge_to_id) and (waypoint1 == edge_from_id):
                    # This edge matches the pair of waypoints! Add it the edge list and continue.
                    return map_pb2.Edge.Id(from_waypoint=waypoint1, to_waypoint=waypoint2)
        return None

    def _on_quit(self):
        """Cleanup on quit from the command line interface."""
        # Sit the robot down + power off after the navigation command is complete.
        if self._powered_on and not self._started_powered_on:
            self._robot_command_client.robot_command(RobotCommandBuilder.safe_power_off_command(),
                                                     end_time_secs=time.time())

    def run(self):
        """Main loop for the command line interface."""
        while True:
            print("""
            Options:
            (1) Get localization state.
            (2) Initialize localization to the nearest fiducial (must be in sight of a fiducial).
            (3) Initialize localization to a specific waypoint (must be exactly at the waypoint)."""

                  """
            (4) List the waypoint ids and edge ids of the map on the robot.
            (5) Upload the graph and its snapshots.
            (6) Navigate to. The destination waypoint id is the second argument.
            (7) Navigate route. The (in-order) waypoint ids of the route are the arguments.
            (8) Navigate to in seed frame. The following options are accepted for arguments: [x, y],
                [x, y, yaw], [x, y, z, yaw], [x, y, z, qw, qx, qy, qz]. (Don't type the braces).
                When a value for z is not specified, we use the current z height.
                When only yaw is specified, the quaternion is constructed from the yaw.
                When yaw is not specified, an identity quaternion is used.
            (9) Clear the current graph.
            (p) Pick Object at waypoint
            (d) Drop Object at waypoint
            (s) search in graph
            (n) Navigate Waypoints
            (sv): save_modified_pkl_file
            (q) Exit.
            """)
            try:
                inputs = input('>')
            except NameError:
                pass
            req_type = str.split(inputs)[0]

            if req_type == 'q':
                self._on_quit()
                break

            if req_type not in self._command_dictionary:
                print("Request not in the known command dictionary.")
                continue
            try:
                cmd_func = self._command_dictionary[req_type]
                cmd_func(str.split(inputs)[1:])
            except Exception as e:
                print(e)


def main(argv):
    """Run the command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-u', '--upload-filepath',
                        help='Full filepath to graph and snapshots to be uploaded.', required=True)
    bosdyn.client.util.add_base_arguments(parser)
    options = parser.parse_args(argv)

    # Setup and authenticate the robot.
    sdk = bosdyn.client.create_standard_sdk('GraphNavClient')
    robot = sdk.create_robot(options.hostname)
    bosdyn.client.util.authenticate(robot)

    graph_nav_command_line = GraphNavInterface(robot, options.upload_filepath)
    lease_client = robot.ensure_client(LeaseClient.default_service_name)
    try:
        with LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
            try:
                graph_nav_command_line.run()
                return True
            except Exception as exc:  # pylint: disable=broad-except
                print(exc)
                print("Graph nav command line client threw an error.")
                return False
    except ResourceAlreadyClaimedError:
        print(
            "The robot's lease is currently in use. Check for a tablet connection or try again in a few seconds."
        )
        return False


if __name__ == '__main__':
    exit_code = 0
    if not main(sys.argv[1:]):
        exit_code = 1
    os._exit(exit_code)  # Exit hard, no cleanup that could block.
