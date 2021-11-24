#!/usr/bin/env python
# coding: utf-8

import sys
ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
if ros_path in sys.path:
    sys.path.remove(ros_path)
import cv2
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')

import pyzed.sl as sl
import math
import numpy as np
import time
import matplotlib.pyplot as plt
from PIL import Image

def check_episode(distance, episode_start, episode_end):
    init_distance = 20
    final_distance = 1.0
    
    if ~np.isnan(distance):
        if distance <= init_distance and distance >= final_distance:
            episode_start = True
            episode_end = False        
        else:
            episode_start = False
            episode_end = True        
    
    return episode_start, episode_end    

def main():
    # Create a Camera object
    zed = sl.Camera()

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE  # Use PERFORMANCE depth mode
    init_params.coordinate_units = sl.UNIT.METER  # Use meter units (for depth measurements)
    init_params.camera_resolution = sl.RESOLUTION.HD720

    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        exit(1)

    # Create and set RuntimeParameters after opening the camera
    runtime_parameters = sl.RuntimeParameters()
    runtime_parameters.sensing_mode = sl.SENSING_MODE.STANDARD  # Use STANDARD sensing mode
    # Setting the depth confidence parameters
    runtime_parameters.confidence_threshold = 100
    runtime_parameters.textureness_confidence_threshold = 100

    # Capture 50 images and depth, then stop
    i = 0
    image = sl.Mat()
    depth = sl.Mat()
    point_cloud = sl.Mat()
    cam = sl.Camera()
    
    mirror_ref = sl.Transform()
    mirror_ref.set_translation(sl.Translation(2.75,4.0,0))
    tr_np = mirror_ref.m
    
    episode = 0
    image_buffer = []
    init_buffer = 0
    
    episode_start = False
    episode_end = True
    count = True
    while True:
        # A new image is available if grab() returns SUCCESS
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            st = time.time()
            # Retrieve left image
            zed.retrieve_image(image, sl.VIEW.LEFT)
            # Retrieve depth map. Depth is aligned on the left image
            zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
            # Retrieve colored point cloud. Point cloud is aligned on the left image.
            zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
                                    
            # Get and print distance value in mm at the center of the image
            # We measure the distance camera - object using Euclidean distance
            distance_matrix = []
            for x in range(200,1200,10):
                for y in range(200,520,20):
                    err, point_cloud_value = point_cloud.get_value(x, y)
                    distance = math.sqrt(point_cloud_value[0] * point_cloud_value[0] +
                                         point_cloud_value[1] * point_cloud_value[1] +
                                         point_cloud_value[2] * point_cloud_value[2])
            
                    if not np.isnan(distance) and not np.isinf(distance):
                        distance_matrix.append(distance)
            if len(distance_matrix) == 0:
                temp_distance = np.nan
            else:
                temp_distance = np.min(distance_matrix)

            # Check whether an episode is end or not
            episode_start, episode_end = check_episode(temp_distance, episode_start, episode_end)
                        
            # Assign image data into a temp memory
            temp_image = image.get_data()
            temp_image = cv2.cvtColor(temp_image, cv2.COLOR_RGBA2RGB)
            temp_depth = depth.get_data()
            # Visualization of point cloud map
            "If you want to see the shape of the point cloud data, use: point_cloud.get_data()"
            
            # See rgb image
            cv2.imshow("RGB", temp_image)
            cv2.waitKey(1)                                    
           
            # See depth image
            cv2.imshow("Depth", depth.get_data())
            cv2.waitKey(1)
        
            image_buffer.append(temp_image)
            init_buffer = len(image_buffer)
            
            if init_buffer > 40 and episode_start and not episode_end:
 
                if count:
                    time.sleep(1.5)
                    image_buffer = []
                count = False
                
                temp_image = image.get_data()
                temp_image = cv2.cvtColor(temp_image, cv2.COLOR_RGBA2RGB)
                temp_image = temp_image[:,160:1120,:]
                image_buffer.append(temp_image)
                print('recording. Current Episode:{}'.format(episode), end='\n')
                
                cv2.imwrite('./training_data/samples/RGB/rgb_' + str(i) + '.jpg', temp_image)
                np.save('./training_data/samples/Depth/depth_' + str(i) + '.npy', temp_depth)
                time.sleep(0.5)
                
                i = i + 1
                if not np.isnan(temp_distance) and not np.isinf(temp_distance):
                    print("Distance to camera:{}".format(temp_distance))
                                
            elif init_buffer > 40 and not episode_start and episode_end:
                print('Not recording \n')
                if not count:
                    # Store the image buffer in the training folder
#                     np.save('./training_data/test_images/episode_' + str(episode) + '.npy', image_buffer)
                    # Add episode number an initialize all buffers
                    episode = episode + 1
                    count = True
                    image_buffer = [] 
                    init_buffer = []
            sys.stdout.flush()
    # Close the camera
    zed.close()

if __name__ == "__main__":
    main()


