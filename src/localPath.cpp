#include <iostream>
#include <ros/console.h> 
#include <ros/ros.h>
#include <move_base_msgs/MoveBaseAction.h>
#include <actionlib/client/simple_action_client.h>

#include <nav_msgs/Path.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Twist.h>
#include <tf/transform_listener.h>
//class template
class RLNavNode{

    public:
        RLNavNode();
        int get_thread_numbers();

    private:
        //node handle
        ros::NodeHandle _nh;
        tf::TransformListener _tf_listener;
        //Publisher, Subscribers names
        ros::Subscriber sub_odom, _sub_path, _sub_goal, _sub_amcl;
        ros::Publisher _pub_odompath;

        //private variable
        geometry_msgs::Point _goal_pos;

        //Topics names
        std::string _globalPath_topic, _goal_topic;
        std::string _map_frame, _odom_frame, _car_frame;
        nav_msgs::Path _odom_path;
        bool _goal_received, _goal_reached, _path_computed;

        int _controller_freq, _downSampling, _thread_numbers;
        double _pathLength, _goalRadius, _waypointsDist;
        
        //Callback funtion definitions
        void pathCB(const nav_msgs::Path::ConstPtr &pathMsg);
        void goalCB(const geometry_msgs::PoseStamped::ConstPtr &goalMsg);
}; //endof class


//constructor of class
RLNavNode::RLNavNode(){

    //Private parameters handler in this scope
    ros::NodeHandle pn("~");
    _goal_received = false;
    _goal_reached = false;
    _path_computed = false;

    //Pass all the parameters for topics & Frame name
    pn.param("thread_numbers", _thread_numbers, 2);
    pn.param("waypoints_dist", _waypointsDist, -1.0); // unit: m
    pn.param("path_length", _pathLength, 5.0); // unit: m
    pn.param("goal_radius", _goalRadius, 0.5); // unit: m
    pn.param<std::string>("odom_frame", _odom_frame, "odom");
    pn.param<std::string>("map_frame", _map_frame, "map");
    // caries the portion of the global plan that the local planner is currently attempting to follow.
    pn.param<std::string>("global_path_topic", _globalPath_topic, "/move_base/TrajectoryPlannerROS/global_plan"); 
    pn.param<std::string>("goal_topic", _goal_topic, "/move_base_simple/goal");


    //Publishers and Subscribers
    _sub_path = _nh.subscribe(_globalPath_topic, 1, &RLNavNode::pathCB, this);
    _sub_goal = _nh.subscribe(_goal_topic, 1, &RLNavNode::goalCB, this);

    _pub_odompath  = _nh.advertise<nav_msgs::Path>("/rlgoal_reference", 1); // reference path for rl agent  
};

// Public: return _thread_numbers
int RLNavNode::get_thread_numbers()
{
    return _thread_numbers;
}
// Callback: update goal status when receive goal topics
void RLNavNode::goalCB(const geometry_msgs::PoseStamped::ConstPtr &goalMsg)
{
    _goal_pos = goalMsg->pose.position; // write goal x,y position
    _goal_received = true; 
    _goal_reached = false; // set goal reach false
    ROS_INFO("Goal Received :goalCB!");

}
// CallBack: Update path waypoints (conversion to odom frame) when receive global_plan
void RLNavNode::pathCB(const nav_msgs::Path::ConstPtr &pathMsg) //pathMsg is a pointer
{
    
    if (_goal_received && !_goal_reached) //check if reach goal
    {
        nav_msgs::Path odom_path = nav_msgs::Path();

        try
        {
            double total_length = 0.0;
            int sampling = _downSampling;

            //find waypoints distance
            if (_waypointsDist <= 0.0)
            {
                double dx = pathMsg->poses[1].pose.position.x - pathMsg->poses[0].pose.position.x; // different in x
                double dy = pathMsg->poses[1].pose.position.y - pathMsg->poses[0].pose.position.y; // different in x
                _waypointsDist = sqrt(dx * dx + dy * dy); //eucledean distance
                _downSampling = int(_pathLength / 10.0 / _waypointsDist); //downsampling the path to waypoints
                ROS_INFO("waypointdist  %f ", _waypointsDist);
                ROS_INFO("Down sampling %d ", _downSampling);
            }

            // Cut and downsampling the path
            for (int i = 0; i < pathMsg->poses.size(); i++)
            {
                if (total_length > _pathLength) //check if greater than the path Length already?
                    break;
                    
                ROS_INFO(" sampling %d ", sampling);
                if (sampling == _downSampling)
                {
                    geometry_msgs::PoseStamped tempPose;
                    _tf_listener.transformPose(_odom_frame, ros::Time(0), pathMsg->poses[i], _map_frame, tempPose);
                    odom_path.poses.push_back(tempPose); // add new elements at the end of vector
                    sampling = 0;
                }
                total_length = total_length + _waypointsDist;
                sampling = sampling + 1;
            }

            if (odom_path.poses.size() >= 6)
            {
                _odom_path = odom_path; // Path waypoints in odom frame
                _path_computed = true;
                // publish odom path
                odom_path.header.frame_id = _odom_frame;
                odom_path.header.stamp = ros::Time::now();
                _pub_odompath.publish(odom_path);
                ROS_INFO("Publish path from odom frame :pathCB!");
            }
            else
            {
                ROS_INFO("Failed to path generation %d", odom_path.poses.size() );
                _waypointsDist = -1;
            }
            //DEBUG
            //cout << endl << "N: " << odom_path.poses.size() << endl
            //<<  "Car path[0]: " << odom_path.poses[0];
            // << ", path[N]: " << _odom_path.poses[_odom_path.poses.size()-1] << endl;
        }
        catch (tf::TransformException &ex) //exception handling
        {
            ROS_ERROR("%s", ex.what());
            ros::Duration(1.0).sleep();
        }
    }
}


int main(int argc, char **argv)
{
    ros::init(argc, argv, "RL_NavNode");
    RLNavNode rl_nav;

    ROS_INFO("Waiting for global path msgs ~");
    
    ros::AsyncSpinner spinner(rl_nav.get_thread_numbers()); // Use multi threads
    spinner.start();
    ros::waitForShutdown();
    return 0;
}
