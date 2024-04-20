# Copyright (c) 2022 Boston Dynamics, Inc.  All rights reserved.
#
# Downloading, reproducing, distributing or otherwise using the SDK Software
# is subject to the terms and conditions of the Boston Dynamics Software
# Development Kit License (20191101-BDSDK-SL).

"""Command line interface integrating options to record maps with WASD controls. """
import argparse
import logging
import os
import sys
import time
import cv2
import numpy as np
import tkinter as tk
from tkinter import simpledialog

import google.protobuf.timestamp_pb2
import graph_nav_util
import grpc
from google.protobuf import wrappers_pb2 as wrappers
from visualization_wrapper.class_defs import Nodes, Semantic_location, Object
from geometry_msgs.msg import Pose, Point, Quaternion
from visualization_wrapper.visualiser import Visualiser  # Import the Visualiser class
import pickle


import bosdyn.client.channel
import bosdyn.client.util
from bosdyn.api.graph_nav import map_pb2, map_processing_pb2, recording_pb2
from bosdyn.client import ResponseError, RpcError, create_standard_sdk
from bosdyn.client.graph_nav import GraphNavClient
from bosdyn.client.map_processing import MapProcessingServiceClient
from bosdyn.client.math_helpers import Quat, SE3Pose
from bosdyn.client.recording import GraphNavRecordingServiceClient
from bosdyn.client.image import ImageClient, pixel_to_camera_space, build_image_request
from bosdyn.api import geometry_pb2, image_pb2,manipulation_api_pb2
from bosdyn.client.frame_helpers import get_a_tform_b, ODOM_FRAME_NAME,get_odom_tform_body


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Now try to import segment_image
from grounded_sam_spot_package.grounded_sam_process import segment_image

# g_image_click = []
# g_image_display = []

class RecordingInterface(object):
    """Recording service command line interface."""

    def __init__(self, robot, download_filepath, client_metadata):
        # Keep the robot instance and it's ID.
        self._robot = robot
        # Force trigger timesync.
        self._robot.time_sync.wait_for_sync()

        # Filepath for the location to put the downloaded graph and snapshots.
        if download_filepath[-1] == "/":
            self._download_filepath = download_filepath + "downloaded_graph"
        else:
            self._download_filepath = download_filepath + "/downloaded_graph"

        # Setup the recording service client.
        self._recording_client = self._robot.ensure_client(
            GraphNavRecordingServiceClient.default_service_name)

        # Create the recording environment.
        self._recording_environment = GraphNavRecordingServiceClient.make_recording_environment(
            waypoint_env=GraphNavRecordingServiceClient.make_waypoint_environment(
                client_metadata=client_metadata))

        # Setup the graph nav service client.
        self._graph_nav_client = robot.ensure_client(GraphNavClient.default_service_name)

        self._map_processing_client = robot.ensure_client(
            MapProcessingServiceClient.default_service_name)

        # Store the most recent knowledge of the state of the robot based on rpc calls.
        self._current_graph = None
        self._current_edges = dict()  #maps to_waypoint to list(from_waypoint)
        self._current_waypoint_snapshots = dict()  # maps id to waypoint snapshot
        self._current_edge_snapshots = dict()  # maps id to edge snapshot
        self._current_annotation_name_to_wp_id = dict()

        # Add recording service properties to the command line dictionary.
        self._command_dictionary = {
            '0': self._clear_map,
            '1': self._start_recording,
            '2': self._stop_recording,
            '3': self._get_recording_status,
            '4': self._create_default_waypoint,
            '5': self._download_full_graph,
            '6': self._list_graph_waypoint_and_edge_ids,
            '7': self._create_new_edge,
            '8': self._create_loop,
            '9': self._auto_close_loops_prompt,
            'a': self._optimize_anchoring,
            'o': self._add_object,
            'v': self._visualize_all_nodes
        }
        # camera='frontleft'
        # self._image_source=[camera + '_depth_in_visual_frame',camera + '_fisheye_image']
        self._image_source=['hand_depth_in_hand_color_frame','hand_color_image']
        self._object_cap_dict={
            '0':self._create_waypoint_and_capt_obj_node,
            '1':self._capt_object
        }
        self.g_image_clicks=[]
        self.g_image_display=[]


        self.object_classes=["Bottle", "Clamp","Rubicks Cube","Brush"]
        self.location_classes=["Floor","Tabletop","Shelf"]

        self.thing_classes=self.object_classes+self.location_classes
        self.visualizer=Visualiser()

        self.waypoint_nodes=[]


    def should_we_start_recording(self):
        # Before starting to record, check the state of the GraphNav system.
        graph = self._graph_nav_client.download_graph()
        if graph is not None:
            # Check that the graph has waypoints. If it does, then we need to be localized to the graph
            # before starting to record
            if len(graph.waypoints) > 0:
                localization_state = self._graph_nav_client.get_localization_state()
                if not localization_state.localization.waypoint_id:
                    # Not localized to anything in the map. The best option is to clear the graph or
                    # attempt to localize to the current map.
                    # Returning false since the GraphNav system is not in the state it should be to
                    # begin recording.
                    return False
        # If there is no graph or there exists a graph that we are localized to, then it is fine to
        # start recording, so we return True.
        return True

    def _clear_map(self, *args):
        """Clear the state of the map on the robot, removing all waypoints and edges."""
        return self._graph_nav_client.clear_graph()

    def _start_recording(self, *args):
        """Start recording a map."""
        should_start_recording = self.should_we_start_recording()
        if not should_start_recording:
            print("The system is not in the proper state to start recording.", \
                   "Try using the graph_nav_command_line to either clear the map or", \
                   "attempt to localize to the map.")
            return
        try:
            status = self._recording_client.start_recording(
                recording_environment=self._recording_environment)
            print("Successfully started recording a map.")
        except Exception as err:
            print("Start recording failed: " + str(err))

    def _stop_recording(self, *args):
        """Stop or pause recording a map."""
        first_iter = True
        while True:
            try:
                status = self._recording_client.stop_recording()
                print("Successfully stopped recording a map.")
                break
            except bosdyn.client.recording.NotReadyYetError as err:
                # It is possible that we are not finished recording yet due to
                # background processing. Try again every 1 second.
                if first_iter:
                    print("Cleaning up recording...")
                first_iter = False
                time.sleep(1.0)
                continue
            except Exception as err:
                print("Stop recording failed: " + str(err))
                break

    def _get_recording_status(self, *args):
        """Get the recording service's status."""
        status = self._recording_client.get_record_status()
        if status.is_recording:
            print("The recording service is on.")
        else:
            print("The recording service is off.")

    def _create_default_waypoint(self, *args):
        """Create a default waypoint at the robot's current location."""
        resp = self._recording_client.create_waypoint(waypoint_name="default")
        if resp.status == recording_pb2.CreateWaypointResponse.STATUS_OK:
            print("Successfully created a waypoint.")
        else:
            print("Could not create a waypoint.")

    def _create_custom_waypoint(self,name, *args):
        """Create a custom waypoint at the robot's current location."""
        resp = self._recording_client.create_waypoint(waypoint_name=name)
        if resp.status == recording_pb2.CreateWaypointResponse.STATUS_OK:
            print("Successfully created waypoint- "+name)
        else:
            print("Could not create waypoint- "+name)

    def _download_full_graph(self, *args):
        """Download the graph and snapshots from the robot."""
        graph = self._graph_nav_client.download_graph()
        if graph is None:
            print("Failed to download the graph.")
            return
        self._write_full_graph(graph)
        print("Graph downloaded with {} waypoints and {} edges".format(
            len(graph.waypoints), len(graph.edges)))
        # Download the waypoint and edge snapshots.
        self._download_and_write_waypoint_snapshots(graph.waypoints)
        self._download_and_write_edge_snapshots(graph.edges)
        with open('downloaded_graph/semantic_locations.pkl', 'wb') as file:
            pickle.dump(self.waypoint_nodes, file)

    def _write_full_graph(self, graph):
        """Download the graph from robot to the specified, local filepath location."""
        graph_bytes = graph.SerializeToString()
        self._write_bytes(self._download_filepath, '/graph', graph_bytes)

    def _download_and_write_waypoint_snapshots(self, waypoints):
        """Download the waypoint snapshots from robot to the specified, local filepath location."""
        num_waypoint_snapshots_downloaded = 0
        for waypoint in waypoints:
            if len(waypoint.snapshot_id) == 0:
                continue
            try:
                waypoint_snapshot = self._graph_nav_client.download_waypoint_snapshot(
                    waypoint.snapshot_id)
            except Exception:
                # Failure in downloading waypoint snapshot. Continue to next snapshot.
                print("Failed to download waypoint snapshot: " + waypoint.snapshot_id)
                continue
            self._write_bytes(self._download_filepath + '/waypoint_snapshots',
                              '/' + waypoint.snapshot_id, waypoint_snapshot.SerializeToString())
            num_waypoint_snapshots_downloaded += 1
            print("Downloaded {} of the total {} waypoint snapshots.".format(
                num_waypoint_snapshots_downloaded, len(waypoints)))

    def _download_and_write_edge_snapshots(self, edges):
        """Download the edge snapshots from robot to the specified, local filepath location."""
        num_edge_snapshots_downloaded = 0
        num_to_download = 0
        for edge in edges:
            if len(edge.snapshot_id) == 0:
                continue
            num_to_download += 1
            try:
                edge_snapshot = self._graph_nav_client.download_edge_snapshot(edge.snapshot_id)
            except Exception:
                # Failure in downloading edge snapshot. Continue to next snapshot.
                print("Failed to download edge snapshot: " + edge.snapshot_id)
                continue
            self._write_bytes(self._download_filepath + '/edge_snapshots', '/' + edge.snapshot_id,
                              edge_snapshot.SerializeToString())
            num_edge_snapshots_downloaded += 1
            print("Downloaded {} of the total {} edge snapshots.".format(
                num_edge_snapshots_downloaded, num_to_download))

    def _write_bytes(self, filepath, filename, data):
        """Write data to a file."""
        os.makedirs(filepath, exist_ok=True)
        with open(filepath + filename, 'wb+') as f:
            f.write(data)
            f.close()

    def _update_graph_waypoint_and_edge_ids(self, do_print=False):
        # Download current graph
        graph = self._graph_nav_client.download_graph()
        if graph is None:
            print("Empty graph.")
            return
        self._current_graph = graph

        localization_id = self._graph_nav_client.get_localization_state().localization.waypoint_id

        # Update and print waypoints and edges
        self._current_annotation_name_to_wp_id, self._current_edges = graph_nav_util.update_waypoints_and_edges(
            graph, localization_id, do_print)

    def _list_graph_waypoint_and_edge_ids(self, *args):
        """List the waypoint ids and edge ids of the graph currently on the robot."""
        self._update_graph_waypoint_and_edge_ids(do_print=True)

    def _create_new_edge(self, *args):
        """Create new edge between existing waypoints in map."""

        if len(args[0]) != 2:
            print("ERROR: Specify the two waypoints to connect (short code or annotation).")
            return

        self._update_graph_waypoint_and_edge_ids(do_print=False)

        from_id = graph_nav_util.find_unique_waypoint_id(args[0][0], self._current_graph,
                                                         self._current_annotation_name_to_wp_id)
        to_id = graph_nav_util.find_unique_waypoint_id(args[0][1], self._current_graph,
                                                       self._current_annotation_name_to_wp_id)

        print("Creating edge from {} to {}.".format(from_id, to_id))

        from_wp = self._get_waypoint(from_id)
        if from_wp is None:
            return

        to_wp = self._get_waypoint(to_id)
        if to_wp is None:
            return

        # Get edge transform based on kinematic odometry
        edge_transform = self._get_transform(from_wp, to_wp)

        # Define new edge
        new_edge = map_pb2.Edge()
        new_edge.id.from_waypoint = from_id
        new_edge.id.to_waypoint = to_id
        new_edge.from_tform_to.CopyFrom(edge_transform)

        print("edge transform =", new_edge.from_tform_to)

        # Send request to add edge to map
        self._recording_client.create_edge(edge=new_edge)

    def _create_loop(self, *args):
        """Create edge from last waypoint to first waypoint."""

        self._update_graph_waypoint_and_edge_ids(do_print=False)

        if len(self._current_graph.waypoints) < 2:
            self._add_message(
                "Graph contains {} waypoints -- at least two are needed to create loop.".format(
                    len(self._current_graph.waypoints)))
            return False

        sorted_waypoints = graph_nav_util.sort_waypoints_chrono(self._current_graph)
        edge_waypoints = [sorted_waypoints[-1][0], sorted_waypoints[0][0]]

        self._create_new_edge(edge_waypoints)

    def _auto_close_loops_prompt(self, *args):
        print("""
        Options:
        (0) Close all loops.
        (1) Close only fiducial-based loops.
        (2) Close only odometry-based loops.
        (q) Back.
        """)
        try:
            inputs = input('>')
        except NameError:
            return
        req_type = str.split(inputs)[0]
        close_fiducial_loops = False
        close_odometry_loops = False
        if req_type == '0':
            close_fiducial_loops = True
            close_odometry_loops = True
        elif req_type == '1':
            close_fiducial_loops = True
        elif req_type == '2':
            close_odometry_loops = True
        elif req_type == 'q':
            return
        else:
            print("Unrecognized command. Going back.")
            return
        self._auto_close_loops(close_fiducial_loops, close_odometry_loops)

    def _auto_close_loops(self, close_fiducial_loops, close_odometry_loops, *args):
        """Automatically find and close all loops in the graph."""
        response = self._map_processing_client.process_topology(
            params=map_processing_pb2.ProcessTopologyRequest.Params(
                do_fiducial_loop_closure=wrappers.BoolValue(value=close_fiducial_loops),
                do_odometry_loop_closure=wrappers.BoolValue(value=close_odometry_loops)),
            modify_map_on_server=True)
        print("Created {} new edge(s).".format(len(response.new_subgraph.edges)))

    def _optimize_anchoring(self, *args):
        """Call anchoring optimization on the server, producing a globally optimal reference frame for waypoints to be expressed in."""
        response = self._map_processing_client.process_anchoring(
            params=map_processing_pb2.ProcessAnchoringRequest.Params(),
            modify_anchoring_on_server=True, stream_intermediate_results=False)
        if response.status == map_processing_pb2.ProcessAnchoringResponse.STATUS_OK:
            print("Optimized anchoring after {} iteration(s).".format(response.iteration))
        else:
            print("Error optimizing {}".format(response))

    def cv_mouse_callback(self, event, x, y, flags, param):
        
        clone = self.g_image_display.copy()

        # Handle left button release event
        if event == cv2.EVENT_LBUTTONUP:
            # Prompt for object name
            tk_root = tk.Tk()
            tk_root.withdraw()  # Hide the main window
            object_name = simpledialog.askstring("Input", "Enter object name:", parent=tk_root)
            tk_root.destroy()

            if object_name:
                print("added object name")
                self.g_image_clicks.append({'coords': (x, y), 'name': object_name})
                self._robot.logger.info(f'Object "{object_name}" added at ({x}, {y})')
            else:
                self._robot.logger.info('No object name entered, point not added')

        # Draw lines and markers on the image
        color_line = (30, 30, 30)
        color_marker = (0, 255, 0)
        thickness_line = 2
        thickness_marker = 2
        marker_type = cv2.MARKER_TILTED_CROSS
        image_title = 'Select objects to grasp'
        height, width = clone.shape[:2]

        # Draw current mouse position lines
        cv2.line(clone, (0, y), (width, y), color_line, thickness_line)
        cv2.line(clone, (x, 0), (x, height), color_line, thickness_line)

        # Draw markers for previously clicked points and annotate with names
        for point_info in self.g_image_clicks:
            point = point_info['coords']
            cv2.drawMarker(clone, point, color_marker, marker_type, markerSize=10, thickness=thickness_marker)
            cv2.putText(clone, point_info['name'], (point[0] + 10, point[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_marker, 1, cv2.LINE_AA)

        cv2.imshow(image_title, clone)

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

    def _create_waypoint_and_capt_obj_node(self,*args):
        print("capturing waypoint")
        wp_name=input("Enter waypoint name: ")
        self._create_custom_waypoint(wp_name)
        self._capt_object(wp_name)

    def _visualize_all_nodes(self,*args):
        for waypoint_node in self.waypoint_nodes:
            self.visualizer.visualise_node(waypoint_node)

    def _capt_object(self,waypoint_name=None,*args):

        if(waypoint_name==None):
            print("waypoint name not input")
            return

        state = self._graph_nav_client.get_localization_state()       
        odom_tform_body = get_odom_tform_body(state.robot_kinematics.transforms_snapshot)
        
        waypoint_pose=Pose()    
        waypoint_pose.position.x,waypoint_pose.position.y,waypoint_pose.position.z= odom_tform_body.x,odom_tform_body.y,odom_tform_body.z
        waypoint_pose.orientation.x,waypoint_pose.orientation.y,waypoint_pose.orientation.z,waypoint_pose.orientation.w=odom_tform_body.rotation.x, odom_tform_body.rotation.y,odom_tform_body.rotation.z,odom_tform_body.rotation.w
        waypoint_node = Nodes(waypoint_name,waypoint_pose )
        
        print("waypoint position: "+ str(waypoint_pose.position.x)+" "+str(waypoint_pose.position.y)+" "+str(waypoint_pose.position.z))
        #capture objects
        image_client = self._robot.ensure_client(ImageClient.default_service_name)
        image_requests=[]
        for i,source in enumerate(self._image_source):
            image_requests.append(image_pb2.ImageRequest(image_source_name=self._image_source[i],quality_percent=100))

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

        detections,annotated_image=segment_image(visual_rgb,self.thing_classes)
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
            
            depth_point=cv_depth[mean_location[0],mean_location[1]]

            
            tform_snapshot = image_responses[1].shot.transforms_snapshot
            if(min_val_depth<2000): #depth less than 1 meter
                pixel_point=pixel_to_camera_space(image_responses[1],mean_location[0],mean_location[1],depth_point/1000)
                cam_to_world_tform = get_a_tform_b(tform_snapshot,ODOM_FRAME_NAME,image_responses[1].shot.frame_name_image_sensor)
                world_coord=cam_to_world_tform.transform_cloud(pixel_point)
                
                print("added "+str(object_name)+" at "+str(world_coord) +" to the graph")
                temp_pose=Pose()
                temp_pose.position.x,temp_pose.position.y,temp_pose.position.z=world_coord
                temp_pose.orientation.x=1
                annotated_image=cv2.circle(annotated_image,(mean_location[1], mean_location[0]),radius=20,color=(0,0,255),thickness=-1)
                cv2.imshow("segmentations",annotated_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                if object_name in self.location_classes:
                    loc_node= Semantic_location(object_name, temp_pose, waypoint_name,detections.xyxy[i])
                    waypoint_node.add_location(loc_node)
                else:
                    obj_node= Object(object_name,temp_pose,detections.xyxy[i])
                    obj_list.append(obj_node)
                #waypoint.annotations[object_name].CopyFrom(grasp)

        # Upload the modified graph back to the robot
        for obj in obj_list:
            max_inter,max_location,min_dist,closest_loc=0,None,1e7,None
            for semantic_location in waypoint_node.get_locations():
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
        self.visualizer.visualise_node(waypoint_node)
        self.waypoint_nodes.append(waypoint_node)
        #self._graph_nav_client.upload_graph(graph)
        print(f"Successfully added objects.")
        

        return True


    def _add_object(self,*args):
        """A simple example of using the Boston Dynamics API to command Spot's arm."""

        
        status = self._recording_client.get_record_status()
        if status.is_recording:
            print("Recording. Proceed to capture object node")
        else:
            print("Start recording to capture object.")
            return
        
        while True:
            print("""
            Options for capturing object:
            (0) create a new waypoint at spot loaction and capture object.
            (1) add object to previous waypoint. (add waypoint to add to specific waypoint)
            (q) Exit.
            """)
            try:
                inputs = input('>')
            except NameError:
                pass
            req_type = str.split(inputs)[0]

            if req_type == 'q':
                break

            if req_type not in self._object_cap_dict:
                print("Request not in the known command dictionary.")
                continue
            try:
                cmd_func = self._object_cap_dict[req_type]
                cmd_func(str.split(inputs)[1:])
            except Exception as e:
                print(e)



    def _get_waypoint(self, id):
        """Get waypoint from graph (return None if waypoint not found)"""

        if self._current_graph is None:
            self._current_graph = self._graph_nav_client.download_graph()

        for waypoint in self._current_graph.waypoints:
            if waypoint.id == id:
                return waypoint

        print('ERROR: Waypoint {} not found in graph.'.format(id))
        return None

    def _get_transform(self, from_wp, to_wp):
        """Get transform from from-waypoint to to-waypoint."""

        from_se3 = from_wp.waypoint_tform_ko
        from_tf = SE3Pose(
            from_se3.position.x, from_se3.position.y, from_se3.position.z,
            Quat(w=from_se3.rotation.w, x=from_se3.rotation.x, y=from_se3.rotation.y,
                 z=from_se3.rotation.z))

        to_se3 = to_wp.waypoint_tform_ko
        to_tf = SE3Pose(
            to_se3.position.x, to_se3.position.y, to_se3.position.z,
            Quat(w=to_se3.rotation.w, x=to_se3.rotation.x, y=to_se3.rotation.y,
                 z=to_se3.rotation.z))

        from_T_to = from_tf.mult(to_tf.inverse())
        return from_T_to.to_proto()

    def run(self):
        """Main loop for the command line interface."""
        while True:
            print("""
            Options:
            (0) Clear map.
            (1) Start recording a map.
            (2) Stop recording a map.
            (3) Get the recording service's status.
            (4) Create a default waypoint in the current robot's location.
            (5) Download the map after recording.
            (6) List the waypoint ids and edge ids of the map on the robot.
            (7) Create new edge between existing waypoints using odometry.
            (8) Create new edge from last waypoint to first waypoint using odometry.
            (9) Automatically find and close loops.
            (a) Optimize the map's anchoring.
            (o) Add object node
            (v) Visualize all nodes
            (q) Exit.
            """)
            try:
                inputs = input('>')
            except NameError:
                pass
            req_type = str.split(inputs)[0]

            if req_type == 'q':
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
    bosdyn.client.util.add_base_arguments(parser)
    parser.add_argument('-d', '--download-filepath',
                        help='Full filepath for where to download graph and snapshots.',
                        default=os.getcwd())
    parser.add_argument(
        '-n', '--recording_user_name', help=
        'If a special user name should be attached to this session, use this name. If not provided, the robot username will be used.',
        default='')
    parser.add_argument(
        '-s', '--recording_session_name', help=
        'Provides a special name for this recording session. If not provided, the download filepath will be used.',
        default='')
    options = parser.parse_args(argv)

    # Create robot object.
    sdk = bosdyn.client.create_standard_sdk('RecordingClient')
    robot = sdk.create_robot(options.hostname)
    bosdyn.client.util.authenticate(robot)

    # Parse session and user name options.
    session_name = options.recording_session_name
    if session_name == '':
        session_name = os.path.basename(options.download_filepath)
    user_name = options.recording_user_name
    if user_name == '':
        user_name = robot._current_user

    # Crate metadata for the recording session.
    client_metadata = GraphNavRecordingServiceClient.make_client_metadata(
        session_name=session_name, client_username=user_name, client_id='RecordingClient',
        client_type='Python SDK')
    recording_command_line = RecordingInterface(robot, options.download_filepath, client_metadata)

    try:
        recording_command_line.run()
        return True
    except Exception as exc:  # pylint: disable=broad-except
        print(exc)
        print("Recording command line client threw an error.")
        return False


if __name__ == '__main__':
    if not main(sys.argv[1:]):
        sys.exit(1)
