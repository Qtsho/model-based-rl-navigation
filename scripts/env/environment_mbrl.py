#!/usr/bin/env python

# Authors: Tien Tran, adapted from ROBOTIS 
# mail: quang.tran@fh-dortmund.de

import rospy
import numpy as np
import math
from math import pi
from geometry_msgs.msg import Twist, Point, Pose, PoseWithCovarianceStamped
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from tf.transformations import euler_from_quaternion, quaternion_from_euler

# from respawnGoal import Respawn
planner = False
if planner:
    from env.respawnPlannerGoal import Respawn
else:
    from env.respawnGoal import Respawn
    
class Env():
    def __init__(self, action_size):
        self.goal_x = 0
        self.goal_y = 0
        self.heading = 0
        self.action_size = action_size
        self.action_space = np.arange(0,action_size,1)
        self.observation_space = (2,)
        self.max_scan = 5.5

        self.usedAMCL = planner #set this if use AMCL as localization, False will use odometry
        self.initGoal = True
        self.get_goalbox = False


        self.position = Pose()
        self.pub_cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=5)
        self.sub_odom = rospy.Subscriber('odom', Odometry, self.getOdometry) # altomatically jump to callback
        self.sub_pose = rospy.Subscriber ('amcl_pose', PoseWithCovarianceStamped,self.getAMCLPose)
        self.reset_proxy = rospy.ServiceProxy('gazebo/reset_world', Empty)
        self.unpause_proxy = rospy.ServiceProxy('gazebo/unpause_physics', Empty)
        self.pause_proxy = rospy.ServiceProxy('gazebo/pause_physics', Empty)
        self.respawn_goal = Respawn()
        self.min_range_to_reach_goal = 0.01  # goal range
        self.min_range_to_collision = 0.13 #collision range r

        #local goal
        self.local_distance  = 0 
        self.goal_local_x  = 0 
        self.goal_local_y  = 0


    def getGoalDistace(self, goal_x, goal_y): #can be used for local and global goal. an for random position
        goal_distance = round(math.hypot(goal_x - self.position.x, goal_y - self.position.y), 2)
        return goal_distance
        

    def getAMCLPose(self, pose):
        if (self.usedAMCL):
            self.position = pose.pose.pose.position
            orientation = pose.pose.pose.orientation
            orientation_list = [orientation.x, orientation.y, orientation.z, orientation.w] # a list of orientation as quaternion
            _, _, yaw = euler_from_quaternion(orientation_list) # roll, pitch, yaw
            goal_angle = math.atan2(self.goal_y - self.position.y, self.goal_x - self.position.x)

            heading = goal_angle - yaw
            if heading > pi:
                heading -= 2 * pi
            elif heading < -pi:
                heading += 2 * pi
            self.heading = round(heading, 2)
            print('amcl pose:', self.position)

        else:
            print('Not use AMCL, instead only odom, set flag to True to use AMCL localization')

    def getOdometry(self, odom):# call back when odometry data is receivee
        if (not self.usedAMCL):
            self.position = odom.pose.pose.position
            orientation = odom.pose.pose.orientation
            orientation_list = [orientation.x, orientation.y, orientation.z, orientation.w] # a list of orientation as quaternion
            _, _, yaw = euler_from_quaternion(orientation_list)

            goal_angle = math.atan2(self.goal_y - self.position.y, self.goal_x - self.position.x) #todo, local goal, global goal

            heading = goal_angle - yaw
            if heading > pi:
                heading -= 2 * pi

            elif heading < -pi:
                heading += 2 * pi

            self.heading = round(heading, 2)


    def _get_obs(self):
        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('scan', LaserScan, timeout=5)
            except:
                print ("Error while waiting laser message!")
                pass
    
        heading = self.heading # update when ever get odometry
        self.obs_dict = {}
        self.obs_dict['position'] = np.append(self.position, self.heading)
        # current_distance = round(math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y),2)
        current_distance  = self.getGoalDistace(self.goal_x,self.goal_y)
        if (self.usedAMCL):
            self.goal_local_x = self.respawn_goal.local_goal_x #change every new local goal available
            self.goal_local_y = self.respawn_goal.local_goal_y #change every new local goal available
 
        self.local_distance  = self.getGoalDistace(self.goal_local_x, self.goal_local_y)
        done = 0
            
        scan_range = []
        for i in range(len(data.ranges)):
            if data.ranges[i] == float('Inf'):
                scan_range.append(self.max_scan)
            elif np.isnan(data.ranges[i]):
                scan_range.append(0)
            else:
                scan_range.append(data.ranges[i])

        
        if self.min_range_to_collision > min(scan_range) > 0: #obstacle
            done = 1
            print('Reset because of collision')

        if (current_distance <self.min_range_to_reach_goal):#TODO: update only done when reach globl goal
            self.get_goalbox = True
            rospy.loginfo("Goal!!")
            # self.reward_dict['distance reward']  = 1000
            self.pub_cmd_vel.publish(Twist()) # stop
            self.goal_x, self.goal_y = self.respawn_goal.getPosition(True, delete=True)
            self.goal_distance = self.getGoalDistace(self.goal_x,self.goal_y)
            self.get_goalbox = False    
        
     
        observations = np.array([heading, self.local_distance])
        print('Observation [heading2Goal, distance2Goal]:',observations)
        return observations, done

    def step(self, action):
        
        linear = action[0]
        angular = action[1]
        #step
        vel_cmd = Twist()
        vel_cmd.linear.x = linear
        vel_cmd.angular.z = angular
        self.pub_cmd_vel.publish(vel_cmd)
        
        #must wait for the simulation to have next data to ge the next obs:
        data = None
        
        #obs/reward/done/score
        ob, done = self._get_obs()
        rew = self.getReward(ob, action)
        score = 0 #DUMMY, TODO: get score

        #return
        env_info = {'obs_dict': self.obs_dict,
                    'rewards': rew,
                    'score': score}
        return ob, rew, done, env_info

    def getReward(self, observations, actions): 
        
       
        if(len(observations.shape)==1): # 1D array
            observations = np.expand_dims(observations, axis = 0) #covert to 2D array with (1,obs)
            actions = np.expand_dims(actions, axis = 0)
            batch_mode = False
        else:
            batch_mode = True


        headings = observations[:,0]
        yaw_rewards = []
        current_distance = observations [:,1]
        self.reward_dict = {}

        distance_rate = -(2 ** (current_distance / self.goal_distance)) #reward for distance to goal
        if (self.usedAMCL):
            local_distance_rate = -(2 ** (current_distance / self.local_distance))
            self.reward_dict['local distance reward'] = local_distance_rate
        angular_reward= 0

        #TODO: reward funtion
        self.reward_dict['distance reward'] = distance_rate
        
        self.reward_dict['angular reward'] = angular_reward
        # if current_distance:# batch mode, wont work
        #     self.get_goalbox = True
        #     rospy.loginfo("Goal!!")
        #     self.reward_dict['distance reward']  = 1000
        #     self.pub_cmd_vel.publish(Twist()) # stop
        #     self.goal_x, self.goal_y = self.respawn_goal.getPosition(True, delete=True)
        #     self.goal_distance = self.getGoalDistace()
        #     self.get_goalbox = False

        self.reward_dict['r_total'] = self.reward_dict['distance reward']  + self.reward_dict['angular reward'] #+ self.reward_dict['local distance reward'] 

        if(not batch_mode):
            return self.reward_dict['r_total'][0]
        return self.reward_dict['r_total'] 


    def reset(self): # reset when first start or crash 
        rospy.wait_for_service('gazebo/reset_world')
        try:
            self.reset_proxy()
        except (rospy.ServiceException) as e:
            print("gazebo/reset_world service call failed")

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('scan', LaserScan, timeout=5)
            except:
                print ("Error while waiting laser message!")
                pass

        if self.initGoal:
            self.goal_x, self.goal_y = self.respawn_goal.getPosition()
            self.initGoal = False

        self.goal_distance = self.getGoalDistace(self.goal_x,self.goal_y)
        state, done = self._get_obs()

        return state
     

    def pause(self):
        rospy.wait_for_service('gazebo/pause_physics')
        try:
            self.pause_proxy()
        except (rospy.ServiceException) as e:
            print("gazebo/pause_physics service call failed")
    #later added funtion: unpause the simulation
    def unpause (self):
        rospy.wait_for_service('gazebo/unpause_physics')
        try:
            self.unpause_proxy()
        except (rospy.ServiceException) as e:
            print("gazebo/unpause_physics service call failed")

