#!/usr/bin/env python3

# Authors: Tien Tran, adapted from ROBOTIS 
# mail: quang.tran@fh-dortmund.de

import rospy
import random
import time
import os, sys
from gazebo_msgs.srv import SpawnModel, DeleteModel
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Pose

#add movebase goal
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal

class Respawn():

    def __init__(self):
        #goal box model
        self.modelPath = os.path.dirname(os.path.realpath(__file__))
        self.modelPath = self.modelPath.replace('/home/workstation/thesis_ws/src/rl_move_base/scripts',
                                                '/home/workstation/thesis_ws/src/turtlebot3_simulations/turtlebot3_gazebo/models/turtlebot3_square/goal_box/model.sdf')

        
        self.stage = 4
        self.goal_position = Pose()
        self.goal = MoveBaseGoal()
        self.init_goal_x = 0.6
        self.init_goal_y = 0.0
        self.goal_position.position.x = self.init_goal_x
        self.goal_position.position.y = self.init_goal_y
        self.modelName = 'goal'

        self.obstacle_1 = 0.6, 0.6
        self.obstacle_2 = 0.6, -0.6
        self.obstacle_3 = -0.6, 0.6
        self.obstacle_4 = -0.6, -0.6

        self.last_goal_x = self.init_goal_x
        self.last_goal_y = self.init_goal_y
        self.last_index = 0
        self.sub_model = rospy.Subscriber('gazebo/model_states', ModelStates, self.checkModel)
        # Create an action client called "move_base" with action definition file "MoveBaseAction"
        self.actionClient = actionlib.SimpleActionClient('move_base',MoveBaseAction)

        self.check_model = False
        self.index = 0
       

        while True:
            try:
                self.f = open(self.modelPath, 'r')
                break
            except OSError:
                print('cannot open', self.modelPath)
                
        self.model = self.f.read()

    def checkModel(self, model):
        self.check_model = False
        for i in range(len(model.name)):
            if model.name[i] == "goal":
                self.check_model = True

    def respawnModel(self):
        while True:
            if not self.check_model:
                rospy.wait_for_service('gazebo/spawn_sdf_model')
                spawn_model_prox = rospy.ServiceProxy('gazebo/spawn_sdf_model', SpawnModel)
                spawn_model_prox(self.modelName, self.model, 'robotos_name_space', self.goal_position, "world")
                rospy.loginfo("Goal position : %.1f, %.1f", self.goal_position.position.x,
                              self.goal_position.position.y)
                break
            else:
                pass
    #delete the goal box model
    def deleteModel(self): 
        while True:
            if self.check_model:
                rospy.wait_for_service('gazebo/delete_model')
                del_model_prox = rospy.ServiceProxy('gazebo/delete_model', DeleteModel)
                del_model_prox(self.modelName)
                break
            else:
                pass
    
    def getPosition(self, position_check=False, delete=False):
        if delete:
            self.deleteModel()

        if self.stage != 4:
            while position_check: # check not to repeat the goal at the same position
                goal_x = random.randrange(-12, 13) / 10.0
                goal_y = random.randrange(-12, 13) / 10.0
                if abs(goal_x - self.obstacle_1[0]) <= 0.4 and abs(goal_y - self.obstacle_1[1]) <= 0.4:
                    position_check = True
                elif abs(goal_x - self.obstacle_2[0]) <= 0.4 and abs(goal_y - self.obstacle_2[1]) <= 0.4:
                    position_check = True
                elif abs(goal_x - self.obstacle_3[0]) <= 0.4 and abs(goal_y - self.obstacle_3[1]) <= 0.4:
                    position_check = True
                elif abs(goal_x - self.obstacle_4[0]) <= 0.4 and abs(goal_y - self.obstacle_4[1]) <= 0.4:
                    position_check = True
                elif abs(goal_x - 0.0) <= 0.4 and abs(goal_y - 0.0) <= 0.4:
                    position_check = True
                else:
                    position_check = False

                if abs(goal_x - self.last_goal_x) < 1 and abs(goal_y - self.last_goal_y) < 1:
                    position_check = True

                self.goal_position.position.x = goal_x
                self.goal_position.position.y = goal_y

        else:
            while position_check:
                goal_x_list = [0.6, 1.9, 0.5, 0.2, -0.8, -1, -1.9, 0.5, 2, 0.5, 0, -0.1, -2]
                goal_y_list = [0, -0.5, -1.9, 1.5, -0.9, 1, 1.1, -1.5, 1.5, 1.8, -1, 1.6, -0.8]

                self.index = random.randrange(0, 13)
                print(self.index, self.last_index)
                if self.last_index == self.index:
                    position_check = True
                else:
                    self.last_index = self.index
                    position_check = False

                self.goal_position.position.x = goal_x_list[self.index]
                self.goal_position.position.y = goal_y_list[self.index]

        time.sleep(0.5)
        self.respawnModel()

        self.last_goal_x = self.goal_position.position.x
        self.last_goal_y = self.goal_position.position.y
        
        self.goal.target_pose.header.frame_id = "map"
        self.goal.target_pose.header.stamp = rospy.Time.now()
        self.goal.target_pose.pose.position.x = self.goal_position.position.x
        self.goal.target_pose.pose.position.y = self.goal_position.position.y
        self.goal.target_pose.pose.orientation.w = 1

        #return self.goal_position.position.x, self.goal_position.position.y
        return self.goal

    def sendAction(self):

        while not self.actionClient.wait_for_server():
            print ("Waiting for movebase")
        #we'll send a goal to the robot 
       
        self.actionClient.send_goal(self.getPosition(position_check=True, delete=True))
        wait = self.actionClient.wait_for_result()

        if not wait:
            rospy.logerr("Action server not available!")
            rospy.signal_shutdown("Action server not available!")# turn off signal
        else: 
            state = self.actionClient.get_state()

        if state == 3:
            print ('Goal Reach!!') #TODO check goal reach?
            self.actionClient.send_goal(self.getPosition(position_check=True, delete=True))
            return
        else:
            print('Failed!, sending another goal')
            self.actionClient.send_goal(self.getPosition(position_check=True, delete=True))
            return
        

if __name__ == '__main__':
    try:
        
        rospy.init_node('goalPub', anonymous=True)
        

        respawn_goal = Respawn()
        while True:     
            respawn_goal.sendAction()
            # 

        #rospy.spin()      # if has callback will call as fast as possible     

    except rospy.ROSInterruptException:
        rospy.loginfo("Closing node")       