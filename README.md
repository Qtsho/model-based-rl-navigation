# MBRL-Navigation Planner

A model based RL approach for robot navigation.
## Table of contents
* [General Info](#general-info)
* [Technologies](#technologies)
* [Setup](#setup)

## General Info

Abstract


## Technologies
Project is created with:
* PyTorch
* Python 3.7
* Gazebo
* ROS Melodic

## Setup
In other to start trainning, it is reccommended to have a GPU. As we want to boost the simulation faster with constant rate.

To run this project we need: 
Dependencies:
  ros-rl-env
 
Prequisitive:
  Turtlebot 3 simulation
  Navigation stack for turtlebot
  
run: 

1/ the navigation rl environment: turtlebot3 simulation, preferably stage 4

2/ rosrun rl_move_base turtlebot3_mbrl.py
