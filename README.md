# MBRL-Navigation Planner

A model based RL approach for robot navigation.
## Table of contents
* [General Info](#general-info)
* [Technologies](#technologies)
* [Setup](#setup)

## General Info

This implemented the MBRL algorithms in paper for robot navigation: Neural Network Dynamics for Model-Based Deep Reinforcement Learning with Model-Free Fine-Tuning, Anusha Nagabandi, Gregory Kahn, Ronald S. Fearing, Sergey Levine

https://arxiv.org/abs/1708.02596




## Technologies
Project is created with:
* PyTorch
* Python 3.7
* Gazebo
* ROS Melodic

## Setup

![image](https://user-images.githubusercontent.com/37075262/140490319-3b7a85f4-a7e4-46c1-83a3-b49f4d0b2511.png)

![image](https://user-images.githubusercontent.com/37075262/140488759-ef9c4c35-7800-4687-921e-4bdf6858a9d1.png)

In other to start trainning, it is reccommended to have a GPU. As we want to boost the simulation faster with constant rate.

To run this project we need: 
Dependencies:
  ros-rl-env
 
Prequisitive:
  Turtlebot 3 simulation
  Navigation stack for turtlebot
  
 Branch:
* refractor: code cleannig
* arws: run on workstation

Rename the package folder to rl_move_base to match with package.xml

You have to rename the PATH names in agent.py to match your real path

run: 

1/ the navigation rl environment: turtlebot3 simulation, preferably stage 4
```
$ roslaunch turtlebot3_simulation turtlebot3_stage_$
```

2/ run the MBRL implemetation
```
$ rosrun rl_move_base turtlebot3_mbrl.py
```
