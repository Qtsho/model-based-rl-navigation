#include <ros/ros.h>
#include <move_base_msgs/MoveBaseAction.h>
#include <actionlib/client/simple_action_client.h>

typedef actionlib::SimpleActionClient<move_base_msgs::MoveBaseAction> MoveBaseClient;



int main(int argc, char** argv){
  ros::init(argc, argv, "simple_navigation_goals");
  ros::NodeHandle n;
  //ros::Subscriber goalsub = n.subscribe("goal", 1000, goalCallback);


  //tell the action client that we want to spin a thread by default
  MoveBaseClient actionClient("move_base", true); //- no spin needed when true

  //wait for the action server to come up
  while(!actionClient.waitForServer(ros::Duration(5.0))){
    ROS_INFO("Waiting for the move_base action server to come up");
  }

  move_base_msgs::MoveBaseGoal goal;

  //we'll send a goal to the robot to move 1 meter forward
  goal.target_pose.header.frame_id = "map";
  goal.target_pose.header.stamp = ros::Time::now();
  goal.target_pose.pose.position.x = 0.6;
  goal.target_pose.pose.orientation.w = 1.0;


  ROS_INFO("Sending goal");
  actionClient.sendGoal(goal);
  
  actionClient.waitForResult();

  if(actionClient.getState() == actionlib::SimpleClientGoalState::SUCCEEDED)
    ROS_INFO("Reach Goal!!");
  else
    ROS_INFO("Failed");
  
  return 0;
}
