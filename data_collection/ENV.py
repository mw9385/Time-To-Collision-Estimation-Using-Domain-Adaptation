#!/usr/bin/env python
# coding: utf-8

###########################################################################
###########################################################################
from __future__ import print_function
from geometry_msgs.msg import Twist, Pose
from sensor_msgs.msg import LaserScan, Image
from cv_bridge import CvBridge, CvBridgeError
from keras.utils import to_categorical
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState

import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import random
import rospy
import std_msgs.msg 
import sys, select, termios, tty
import time
###########################################################################
###########################################################################

class ENV(object):
    def __init__(self):        
        # define gazebo connection and import gazebo msgs
        self.cmd_pub = rospy.Publisher('cmd_vel', Twist, queue_size = 10)
        self.pose_pub = rospy.Publisher('/command/pose', Pose, queue_size = 10)        
        self.g_set_state = rospy.ServiceProxy('gazebo/set_model_state',SetModelState)        
              
        self.twist = Twist()
        self.pose = Pose()                       
        self.linear_table = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] 
        self.side_table = [0.2, 0.1, 0, -0.1, -0.2]
        self.state = ModelState()
        self.state.model_name = 'quadrotor'
        self.state.reference_frame = 'world'        
                        
        
    def stop(self):
        self.twist.linear.x = 0; self.twist.linear.y = 0; self.twist.linear.z = 0
        self.twist.angular.x = 0; self.twist.angular.y = 0; self.twist.angular.z = 0                
        self.cmd_pub.publish(self.twist)               
        
    def hovering(self):
             
        self.state.pose.position.x = 0
        self.state.pose.position.y = 0
        self.state.pose.position.z = 1.2
        self.state.pose.orientation.x = 0
        self.state.pose.orientation.y = 0
        self.state.pose.orientation.z = 0
        self.state.twist.linear.x = 0
        self.state.twist.linear.y = 0
        self.state.twist.linear.z = 0
        self.state.twist.angular.x = 0
        self.state.twist.angular.y = 0
        self.state.twist.angular.z = 0

        ret = self.g_set_state(self.state)
        
        
    def Control(self,action, side_action):
        if action < 7:
            self.self_speed[0] = self.linear_table[action]
            self.self_speed[1] = 0
        else:                    
            self.self_speed[1] = self.linear_table[action]

        self.twist.linear.x = self.self_speed[0]
        self.twist.linear.y = self.side_table[side_action]
        self.twist.linear.z = 0.
        self.twist.angular.x = 0.
        self.twist.angular.y = 0.
        self.twist.angular.z = 0.
        self.cmd_pub.publish(self.twist)                
    
    def euler_to_quaternion(self, yaw, pitch, roll):

        qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
        qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
        qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
        qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)

        return [qx, qy, qz, qw]
    
    def quaternion_to_euler(self, x, y, z, w):
        
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        X = math.degrees(math.atan2(t0, t1))

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        Y = math.degrees(math.asin(t2))

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        Z = math.degrees(math.atan2(t3, t4))

        return X, Y, Z
    
    def reset_sim(self, pose, orientation):                
        pose_x = pose.x
        pose_y = pose.y
        pose_z = pose.z
           
#        spawn_table = [[-55.691116,26.98],[-41.73, -24.59], [-36.07, 5.72],[-29.25, -15.33], [-13.58, -16.61], 
#                       [-0.21, -17.23],[13.12, -17.05], [28.01, -17.75], [0.63, 10.13], [1.66, 26.20],
#                       [11.27, 26.33], [48.75, -8.05], [55.85, 16.71]] # engineering 1F
        spawn_table = [[-24.893625,-4.115987], [-27.888660, -15.635707],[0.313832, 0.313832],[-9.559809,12.199582], 
                      [5.583409,13.319511], [24.165285,-7.167769], [22.356915,-15.299837], [30.804581, 1.072067]] 
                       # engineering 2F
#         spawn_table = [[22.083181,-7.589279], [-24.972904, 9.559667], [-27.580791, -5.404228], [-11.248223, -5.52603],
#                       [4.225587, 2.766444], [23.352060,3.823326], [-0.456676, 8.845896], [-25.694090, 2.405492]] # engineering 815

#         spawn_table = [[-26.698088, 8.852719], [-27.668938, 2.162856], [-26.773634,-5.992571], [-6.341585, 9.392965], [-7.148545,2.812936], [3.682608, -6.172786], [12.985077,3.015917], [22.233866, 4.119962], [19.290451, -7.25]]
        
#         spawn_table = [[-33, 1.8], [-28, -16], [-9.5, 11.6], [8, -14], [25, -5]]
        
        rand_indx = np.random.randint(8)        
        ori_x = orientation.x
        ori_y = orientation.y
        ori_z = orientation.z        
        ori_w = orientation.w
                      
        yaw = np.random.randint(-180,180) * np.pi/180
        [ori_x, ori_y, ori_z, ori_w] = self.euler_to_quaternion(yaw, 0, 0)        
                
        self.state.pose.position.x = spawn_table[rand_indx][0] + np.random.randint(-1,1)
        self.state.pose.position.y = spawn_table[rand_indx][1] + np.random.randint(-1,1)
        self.state.pose.position.z = 1.8
        self.state.pose.orientation.x = ori_x
        self.state.pose.orientation.y = ori_y
        self.state.pose.orientation.z = ori_z
        self.state.pose.orientation.w = ori_w
        self.state.twist.linear.x = 0
        self.state.twist.linear.y = 0
        self.state.twist.linear.z = 0
        self.state.twist.angular.x = 0
        self.state.twist.angular.y = 0
        self.state.twist.angular.z = 0        
        self.self_speed = [0.03, 0.0]
        ret = self.g_set_state(self.state)    
        rospy.sleep(0.5)
