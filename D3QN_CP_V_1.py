#!/usr/bin/env python
# coding: utf-8

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import cv2

from ENV import ENV
from keras import backend as K
from keras.optimizers import RMSprop, Adam
from keras.layers.convolutional import Conv2D
from keras.utils import to_categorical
from keras.models import Sequential ,load_model, Model
from keras.backend.tensorflow_backend import set_session
from keras.layers import Input, Dense, Flatten, Lambda, add
from sensor_msgs.msg import LaserScan, Image, Imu
from gazebo_msgs.msg import ModelState, ModelStates
from gazebo_msgs.srv import SetModelState
from geometry_msgs.msg import Vector3Stamped, Twist
from skimage.transform import resize
from skimage.color import rgb2gray
from collections import deque
from PIL import Image as iimage

import matplotlib
import matplotlib.pyplot as plt
import rospy
import tensorflow as tf
import scipy.misc
import numpy as np
import random
import time
import random
import pickle
import copy

laser = None
velocity = None
vel = None
theta = None
pose = None
orientation = None
image = None
depth_img = None

def DepthCallBack(img):
    global depth_img
    depth_img = img.data

def callback_laser(msg):    
    global laser
    laser = msg    
    laser = laser.ranges 
    
def callback_camera(msg):
    global image
    image = np.frombuffer(msg.data, dtype=np.uint8)    
    image = np.reshape(image, [480,640,3]) 
    image = np.array(image)

def GetDepthObservation(image):
    width = 304
    height = 228
    
    test_image = iimage.fromarray(image,'RGB')  
    test_image = test_image.resize([width, height], iimage.ANTIALIAS)
    test_image = np.array(test_image).astype('float32')
    test_image = np.expand_dims(np.asarray(test_image), axis = 0)
    pred = cnn_sess.run(net.get_output(), feed_dict={input_node: test_image})
    pred = np.reshape(pred, [128,160])    
    pred = np.array(pred, dtype=np.float32)
        
    pred[np.isnan(pred)] = 5.
    pred = pred / 3.5
    pred[pred > 1.0] = 1.0     
        
    return pred       

def callback_laser(msg):    
    global laser
    laser = msg    
    laser = laser.ranges 

def crash_check(laser_data, velocity, theta, delta_depth):    
    laser_sensor = np.array(laser_data)
    laser_index = np.isinf(laser_sensor)
    laser_sensor[laser_index] = 30
    laser_sensor = np.array(laser_sensor[300:800])    
    done = False
    vel_flag = False
    zone_1_flag = False    
    
    crash_reward = 0
    depth_reward = 0
    vel_reward = 0
    depth_value = (np.min(laser_sensor) - 0.5) / 2.0       
    
    # reward for zone 1
    if depth_value >= 0.4:         
        depth_reward = 0.4
        vel_flag = True        
        
    # reward for zone 2
    else:                                
        vel_factor = np.absolute(np.cos(velocity))
        _depth_reward = depth_value * vel_factor + delta_depth
        depth_reward = np.min([0.4, _depth_reward])           
        vel_flag = False        
        
    # reward for crash
    if np.min(laser_sensor) <= 0.6:
        done = True                
        vel_flag = False
    
    # reward for velocity
    else:
        if vel_flag:
            vel_reward = velocity * np.cos(theta)* 0.2                
            
        else:
            vel_reward = 0            
    # select reward
    if done:
        reward = -1.0
    else:
        reward = depth_reward + vel_reward  
    
    # for collision probability
    if done:
        CP = 0    
    else:
        _CP = (-0.6 + np.min(laser_sensor)) / velocity        
        CP = np.min([_CP, 20])
        
    return done, reward, np.min(laser_sensor), depth_value, CP


def depth_change(depth,_depth):
    laser = depth   # current depth
    _laser = _depth # previous depth  
    eta = 0.2
    
    delta_depth = eta * np.sign(laser - _laser)
    return delta_depth
    
def show_figure(image):
    #show image using cv2    
#     image = cv2.resize(image, (256*2, 320*2), interpolation=cv2.INTER_CUBIC)
    cv2.imshow('input image', image)    
    cv2.waitKey(1)
    
    
def state_callback(msg):
    global velocity, pose, orientation, vel, theta
    idx = msg.name.index("quadrotor")      
    
    pose = msg.pose[idx].position
    orientation = msg.pose[idx].orientation    
    vel = msg.twist[idx]
    
    velocity_x = vel.linear.x
    velocity_y = vel.linear.y    
    velocity = np.sqrt(velocity_x**2 + velocity_y**2)    
    theta = vel.angular.z

if __name__ == '__main__':
    # Check the gazebo connection
    rospy.init_node('env', anonymous=True)
    
    # Class define
    env = ENV()   
    # Parameter setting for the simulation
    EPISODE = 100000    
    global_step = 0 
    # Observe
    rospy.Subscriber('/camera/rgb/image_raw', Image, callback_camera,queue_size = 5)    
    rospy.Subscriber("/scan", LaserScan, callback_laser,queue_size = 5)
    rospy.Subscriber('gazebo/model_states', ModelStates, state_callback, queue_size= 5)      
    rospy.Subscriber('/camera/depth/image_raw', Image, DepthCallBack,queue_size = 5)
       
    # define command step
    rospy.sleep(2.)             
    rate = rospy.Rate(5)
    env.reset_sim(pose, orientation) 
    
    # define episode and image_steps
    e = 0
    image_steps = 0
    
    
    while e < EPISODE and not rospy.is_shutdown():
        e = e + 1        
        env.reset_sim(pose, orientation)                        
        laser_distance = np.stack((0, 0))        
        delta_depth = 0
        step, score  = 0. ,0.
        done = False                    
                        
        while not done and not rospy.is_shutdown():  
            # wait for service
            rospy.wait_for_message('/camera/rgb/image_raw', Image)
            rospy.wait_for_message('/camera/depth/image_raw', Image)
            rospy.wait_for_message('/gazebo/model_states', ModelStates)
            rospy.wait_for_message('/scan', LaserScan)           
                
            global_step = global_step + 1             
            step = step + 1                       
            
            # Observe: get_reward
            [done, reward, _depth, depth_value, CP] = crash_check(laser, velocity, theta, delta_depth)   
            delta_depth = depth_change(laser_distance[0], laser_distance[1])                                            
            
            next_distance = _depth                      
            laser_distance = np.append(next_distance, laser_distance[0])

            if step > 50 and image_steps <= 70000:
                cv2.imwrite('./training_data/trainB/' + str(image_steps) + '_B.jpg', image)
                print(image_steps)
                image_steps = image_steps + 1                                                                             
            
            if step >= 2000:
                done = True           
                
            rate.sleep()             

