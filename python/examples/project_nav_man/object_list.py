import argparse
import logging
import math
import os
import sys
import time

import google.protobuf.timestamp_pb2
import graph_nav_util
import grpc

import bosdyn.client.channel
import bosdyn.client.util
from bosdyn.api import geometry_pb2, power_pb2, robot_state_pb2
from bosdyn.api.graph_nav import graph_nav_pb2, map_pb2, nav_pb2
from bosdyn.client.exceptions import ResponseError
from bosdyn.client.frame_helpers import get_odom_tform_body
from bosdyn.client.graph_nav import GraphNavClient
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive, ResourceAlreadyClaimedError
from bosdyn.client.math_helpers import Quat, SE3Pose
from bosdyn.client.power import PowerClient, power_on, safe_power_off
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient
from bosdyn.client.robot_state import RobotStateClient

class GraphNavInterface(object):
    """GraphNav service command line interface."""

    def __init__(self,  upload_path):
        # Create the client for the Graph Nav main service.
        if upload_path[-1] == "/":
            self._upload_filepath = upload_path[:-1]
        else:
            self._upload_filepath = upload_path
        self._graph_nav_client = self._robot.ensure_client(GraphNavClient.default_service_name)

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

def main(argv):
    graph_nav_command_line = GraphNavInterface(robot, options.upload_filepath)

if __name__ == '__main__':
    exit_code = 0
    if not main(sys.argv[1:]):
        exit_code = 1
    os._exit(exit_code)  # Exit hard, no cleanup that could block.