#include <iostream>
#include <ros/ros.h>
#include <move_base_msgs/MoveBaseAction.h>
#include <actionlib/client/simple_action_client.h>

#include <nav_msgs/Path.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Twist.h>
//class template
class RLNavNode{

    public:
        RLNavNode();
        int get_thread_numbers();

    private:
        //node handle
        ros::NodeHandle _nh;

        //Publisher, Subscribers names
        ros::Subscriber sub_odom, _sub_path, _sub_goal, _sub_amcl;

        //private variable
        geometry_msgs::Point _goal_pos;

        //Topics names
        std::string _globalPath_topic, _goal_topic;
        bool _goal_received, _goal_reached, _path_computed;

        int _controller_freq, _downSampling, _thread_numbers;

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
    pn.param<std::string>("global_path_topic", _globalPath_topic, "/move_base/TrajectoryPlannerROS/global_plan");
    pn.param<std::string>("goal_topic", _goal_topic, "/move_base_simple/goal");

    //Publishers and Subscribers
    _sub_path = _nh.subscribe(_globalPath_topic, 1, &RLNavNode::pathCB, this);
    _sub_goal = _nh.subscribe(_goal_topic, 1, &RLNavNode::goalCB, this);


};

// Public: return _thread_numbers
int RLNavNode::get_thread_numbers()
{
    return _thread_numbers;
}
// Callback: update goal status when receive goal topics
void RLNavNode::goalCB(const geometry_msgs::PoseStamped::ConstPtr &goalMsg)
{

}
// CallBack: Update path waypoints (conversion to odom frame)
void RLNavNode::pathCB(const nav_msgs::Path::ConstPtr &pathMsg)
{
    if (_goal_received && !_goal_reached)
    {
        nav_msgs::Path odom_path = nav_msgs::Path();
        try
        {
            double total_length = 0.0;
            int sampling = _downSampling;

            //find waypoints distance
            if (_waypointsDist <= 0.0)
            {
                double dx = pathMsg->poses[1].pose.position.x - pathMsg->poses[0].pose.position.x;
                double dy = pathMsg->poses[1].pose.position.y - pathMsg->poses[0].pose.position.y;
                _waypointsDist = sqrt(dx * dx + dy * dy);
                _downSampling = int(_pathLength / 10.0 / _waypointsDist);
            }

            // Cut and downsampling the path
            for (int i = 0; i < pathMsg->poses.size(); i++)
            {
                if (total_length > _pathLength)
                    break;

                if (sampling == _downSampling)
                {
                    geometry_msgs::PoseStamped tempPose;
                    _tf_listener.transformPose(_odom_frame, ros::Time(0), pathMsg->poses[i], _map_frame, tempPose);
                    odom_path.poses.push_back(tempPose);
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
            }
            else
            {
                cout << "Failed to path generation" << endl;
                _waypointsDist = -1;
            }
            //DEBUG
            //cout << endl << "N: " << odom_path.poses.size() << endl
            //<<  "Car path[0]: " << odom_path.poses[0];
            // << ", path[N]: " << _odom_path.poses[_odom_path.poses.size()-1] << endl;
        }
        catch (tf::TransformException &ex)
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
