#!/usr/bin/env python

# Authors: Tien Tran, adapted from ROBOTIS 
# mail: quang.tran@fh-dortmund.de

import rospy
import numpy as np
import math
from math import pi
from geometry_msgs.msg import Twist, Point, Pose
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from respawnGoal import Respawn

class Env():
    def __init__(self, action_size):
        self.goal_x = 0
        self.goal_y = 0
        self.heading = 0
        self.action_size = action_size
        self.action_space = np.arange(0,action_size,1)
        self.observation_space = (28,)
        self.max_scan = 5.5
        self.initGoal = True
        self.get_goalbox = False
        self.position = Pose()
        self.pub_cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=5)
        self.sub_odom = rospy.Subscriber('odom', Odometry, self.getOdometry) # altomatically jump to callback
        self.reset_proxy = rospy.ServiceProxy('gazebo/reset_world', Empty)
        self.unpause_proxy = rospy.ServiceProxy('gazebo/unpause_physics', Empty)
        self.pause_proxy = rospy.ServiceProxy('gazebo/pause_physics', Empty)
        self.respawn_goal = Respawn()
        

    def getGoalDistace(self):
        goal_distance = round(math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y), 2)

        return goal_distance
    
    def getOdometry(self, odom):# call back when odometry data is receivee
        self.position = odom.pose.pose.position
        orientation = odom.pose.pose.orientation
        orientation_list = [orientation.x, orientation.y, orientation.z, orientation.w] # a list of orientation as quaternion
        _, _, yaw = euler_from_quaternion(orientation_list)

        goal_angle = math.atan2(self.goal_y - self.position.y, self.goal_x - self.position.x)

        heading = goal_angle - yaw
        if heading > pi:
            heading -= 2 * pi

        elif heading < -pi:
            heading += 2 * pi

        self.heading = round(heading, 2)

    # def getState(self, scan):
    #     scan_range = []
    #     heading = self.heading
    #     min_range = 0.13
    #     done = False

    #     for i in range(len(scan.ranges)):
    #         if scan.ranges[i] == float('Inf'):
    #             scan_range.append(self.max_scan)
    #         elif np.isnan(scan.ranges[i]):
    #             scan_range.append(0)
    #         else:
    #             scan_range.append(scan.ranges[i])

    #     obstacle_min_range = round(min(scan_range), 2)
    #     obstacle_angle = np.argmin(scan_range)
        
    #     if min_range > min(scan_range) > 0:
    #         done = True

    #     current_distance = round(math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y),2)
    #     if current_distance < 0.2:
    #         self.get_goalbox = True
    #     observation = scan_range + [heading, current_distance, obstacle_min_range, obstacle_angle], done
        
    #     observation = [heading, current_distance], done
    #     print ('Observation'+ observation)
        
    #     return observation

    # def setReward(self, state, done, action):
    #     yaw_reward = []
    #     obstacle_min_range = state[-2]
    #     current_distance = state[-3]
    #     heading = state[-4]

    #     for i in range(5):  #reward for heading to goal
    #         angle = -pi / 4 + heading + (pi / 8 * i) + pi / 2
    #         tr = 1 - 4 * math.fabs(0.5 - math.modf(0.25 + 0.5 * angle % (2 * math.pi) / math.pi)[0])
    #         yaw_reward.append(tr)

    #     distance_rate = 2 ** (current_distance / self.goal_distance) #reward for distance to goal

    #     if obstacle_min_range < 0.5: #reward for obstacle avoidance
    #         ob_reward = -5
    #     else:
    #         ob_reward = 0
    #     #TODO: add time to reward
    #     reward = ((round(yaw_reward[action] * 5, 2)) * distance_rate) + ob_reward

    #     if done:
    #         rospy.loginfo("Collision!!")
    #         reward = -500
    #         self.pub_cmd_vel.publish(Twist()) #stop

    #     if self.get_goalbox:
    #         rospy.loginfo("Goal!!")
    #         reward = 1000
    #         self.pub_cmd_vel.publish(Twist()) # stop
    #         self.goal_x, self.goal_y = self.respawn_goal.getPosition(True, delete=True)
    #         self.goal_distance = self.getGoalDistace()
    #         self.get_goalbox = False

        # return reward
    def _get_obs(self):

        self.obs_dict = {}
        self.obs_dict['x'] = self.position.x.copy()
        self.obs_dict['y'] = self.position.y.copy()
        self.obs_dict['yaw'] = self.get_body_com("torso").flat.copy()
        current_distance = round(math.hypot(self.goal_x - self.position.x, self.goal_y - self.position.y),2)

        return np.concatenate([
            self.obs_dict['x'], #9
            self.obs_dict['y'], #9
            self.obs_dict['yaw'], #3
            
        ])
    
    def step(self, action):
        max_angular_vel = 1.5
        ang_vel = ((self.action_size - 1)/2 - action) * max_angular_vel * 0.5
        #step
        vel_cmd = Twist()
        vel_cmd.linear.x = 0.15
        vel_cmd.angular.z = ang_vel
        self.pub_cmd_vel.publish(vel_cmd)

        #obs/reward/done/score
        ob = self._get_obs()
        rew, done = self.get_reward(ob, action)
        score = self.get_score(ob)

        #return
        env_info = {'obs_dict': self.obs_dict,
                    'rewards': self.reward_dict,
                    'score': score}
        return ob, rew, done, env_info

    # def step(self, action):
    #     max_angular_vel = 1.5
    #     ang_vel = ((self.action_size - 1)/2 - action) * max_angular_vel * 0.5
    #     #lin_vel = ((self.action_size - 1)/2 - action) * max_angular_vel * 0.5
    
    #     vel_cmd = Twist()
    #     vel_cmd.linear.x = 0.15
    #     vel_cmd.angular.z = ang_vel
    #     self.pub_cmd_vel.publish(vel_cmd)

    #     data = None
    #     while data is None:
    #         try:
    #             data = rospy.wait_for_message('scan', LaserScan, timeout=5)
    #         except:
    #             print ("Error while waiting laser message!")
    #             pass

    #     state, done = self.getState(data)
    #     reward = self.setReward(state, done, action)
    #     return np.asarray(state), reward, done

    def getReward(self, observations, actions, done): 

        """get reward/s of given (observations, actions) datapoint or datapoints
        Args:
            observations: (batchsize, obs_dim) or (obs_dim,)
            actions: (batchsize, ac_dim) or (ac_dim,)
        Return:
            r_total: reward of this (o,a) pair, dimension is (batchsize,1) or (1,)
            done: True if env reaches terminal state, dimension is (batchsize,1) or (1,)
        """
        #initialize and reshape as needed, for batch mode
     
        self.reward_dict = {} # init the reward dictinary

        if(len(observations.shape)==1):
            observations = np.expand_dims(observations, axis = 0) # add one more dimension (1, obs_dim)
            actions = np.expand_dims(actions, axis = 0) #add one more dimentions to axis 0 ->(1, ac_dim)
            batch_mode = False
        else:
            batch_mode = True
            N = len(observations.shape) # number of action sequences.

        # loop through all the sequences (batch size) at that time step, calculate the reward funtion
        yaw_reward = []
        x = observations[:, 0].copy()
        y = observations[:, 1].copy()
        theta = observations[:, 2].copy()

        current_distances = round(math.hypot(self.goal_x - x, self.goal_y - y),2)


        distance_reward = 2 ** (current_distances / self.goal_distance) # dummy TODO change this similar to set reward.s
        

        zeros = np.zeros((observations.shape[0],)).copy()

        for i in range(5):  #reward for heading to goal
            angle = -pi / 4 + theta + (pi / 8 * i) + pi / 2
            tr = 1 - 4 * math.fabs(0.5 - math.modf(0.25 + 0.5 * angle % (2 * math.pi) / math.pi)[0])
            yaw_reward.append(tr)
        
        reward = ((round(yaw_reward[actions] * 5, 2)) * distance_reward)

        if done:
            rospy.loginfo("Collision!!")
            reward = -500
            self.pub_cmd_vel.publish(Twist()) #stop

        if self.get_goalbox:
            rospy.loginfo("Goal!!")
            reward = 1000
            self.pub_cmd_vel.publish(Twist()) # stop
            self.goal_x, self.goal_y = self.respawn_goal.getPosition(True, delete=True)
            self.goal_distance = self.getGoalDistace()
            self.get_goalbox = False
        self.reward_dict['distance_reward'] = distance_reward
        #total reward.
        self.reward_dict['r_total'] = reward + self.reward_dict['distance_reward'] 
        ##done is NOT always false for this env 
        dones = zeros.copy() #change when reach local path. this is zeros because cheetah is continous task
        
        if(not batch_mode):
            return self.reward_dict['r_total'][0], dones[0]

        return self.reward_dict['r_total'], dones



    def reset(self):
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

        self.goal_distance = self.getGoalDistace()
        state, done = self.getState(data)

        return np.asarray(state)
        #later added funtion:pause the simulation
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

