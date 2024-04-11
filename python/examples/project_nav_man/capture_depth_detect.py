# Copyright (c) 2022 Boston Dynamics, Inc.  All rights reserved.
#
# Downloading, reproducing, distributing or otherwise using the SDK Software
# is subject to the terms and conditions of the Boston Dynamics Software
# Development Kit License (20191101-BDSDK-SL).

"""Example demonstrating capture of both visual and depth images and then overlaying them."""

import argparse
import sys

import cv2
import numpy as np

import bosdyn.client
import bosdyn.client.util
from bosdyn.client.image import ImageClient
from bosdyn.api import image_pb2,gripper_camera_param_pb2

from grounded_sam_spot_package.grounded_sam_process import segment_image


def main(argv):
    # Parse args
    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)

    options = parser.parse_args(argv)


    sources=['hand_depth_in_hand_color_frame','hand_color_image']
    #sources=['hand_depth','hand_color_in_hand_depth_frame']

    # Create robot object with an image client.
    sdk = bosdyn.client.create_standard_sdk('image_depth_plus_visual')
    robot = sdk.create_robot(options.hostname)
    bosdyn.client.util.authenticate(robot)
    image_client = robot.ensure_client(ImageClient.default_service_name)
    #breakpoint()
    # Capture and save images to disk
    
    image_requests=[]
    for i,source in enumerate(sources):
        image_requests.append(image_pb2.ImageRequest(image_source_name=sources[i],quality_percent=100))

    image_responses = image_client.get_image(image_requests)
    
    # Image responses are in the same order as the requests.
    # Convert to opencv images.

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
    
    object_classes=["Door", "Floor","Chair","Bottle"]
    
    # Map depth ranges to color
    
    # cv2.applyColorMap() only supports 8-bit; convert from 16-bit to 8-bit and do scaling
    min_val = np.min(cv_depth)
    max_val = np.max(cv_depth)
    depth_range = max_val - min_val
    depth8 = (255.0 / depth_range * (cv_depth - min_val)).astype('uint8')
    depth8_rgb = cv2.cvtColor(depth8, cv2.COLOR_GRAY2RGB)
    depth_color = cv2.applyColorMap(depth8_rgb, cv2.COLORMAP_JET)


    

    # Add the two images together.
    out = cv2.addWeighted(visual_rgb, 0.5, depth_color, 0.5, 0)


    # Write the image out.
    filename ="hand" + ".jpg"
    cv2.imwrite(filename, out)
    detections=segment_image(visual_rgb,object_classes)

    out_in_range=visual_rgb
    for i in range(len(detections)):
        depth_seg=detections.mask[i]*cv_depth
        mean_depth=np.mean(depth_seg[depth_seg!=0])
        
        #x, y = (detections[i].xyxy[0][0]+detections[i].xyxy[0][2])/2,(detections[i].xyxy[0][1]+detections[i].xyxy[0][3])/2
        object_name = object_classes[detections.class_id[i]]  # Extract object name

        try:
            min_val_depth=np.min(depth_seg[depth_seg!=0])
        except ValueError:
            min_val_depth=0
        print("Object "+str(object_name)+" detected at "+str(i)+" min depth "+str(min_val_depth))
        if(min_val_depth<1500): #depth less than 1 meter
            bool_mask=depth_seg==0
            bool_mask=np.repeat(bool_mask[:,:,np.newaxis],3,axis=2)
            depth_color_seg=depth_color.copy()
            depth_color_seg[bool_mask]=0
            out_in_range=cv2.addWeighted(out_in_range, 0.9, depth_color_seg, 0.5, 0)
            
            #waypoint.annotations[object_name].CopyFrom(grasp)
    cv2.imwrite("hand_in_range.jpg", out_in_range)
    return True


if __name__ == "__main__":
    if not main(sys.argv[1:]):
        sys.exit(1)