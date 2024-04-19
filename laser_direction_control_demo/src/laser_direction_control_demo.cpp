/**
 * @file laser_direction_control_demo.cpp
 * @author Cygnusx (Cygnusx@domain.com)
 * @brief ros雷达引导包，专为19届智能车比赛赛道适配(测试版)
 * @version 0.1
 * @date 2024-03-08
 * 
 * @copyright Copyright (c) 2024
 * 
 */
#include <ros/ros.h>  
#include <sensor_msgs/LaserScan.h>  
#include <sensor_msgs/Imu.h>
#include <nav_msgs/Odometry.h>

#include <geometry_msgs/Twist.h>
#include <tf/transform_datatypes.h>

#include <iostream>  
#include <stdlib.h>
#include <cstdlib>

#include <move_base_msgs/MoveBaseGoal.h>
#include <move_base_msgs/MoveBaseAction.h>
#include <actionlib/client/simple_action_client.h>

#include <chrono>



using namespace tf;
using namespace nav_msgs;
using namespace actionlib;
using namespace move_base_msgs;
using namespace geometry_msgs;
using namespace ros;
using namespace std;
using namespace sensor_msgs;
using namespace geometry_msgs;
using namespace std::chrono;


/**
 * @brief 定义一个SimpleActionClient，用来给move_base一个目标点
 */
typedef actionlib::SimpleActionClient<move_base_msgs::MoveBaseAction> MoveBaseClient;

geometry_msgs::Twist motion;
static int mission_num=0;
static int back_num = 999;
float df,dl,dr,db;
float odom_x,odom_y,odom_yaw;
void odomCallback(const nav_msgs::OdometryConstPtr odom_msg){
    odom_x = odom_msg->pose.pose.position.x;
    odom_y = odom_msg->pose.pose.position.y;
    odom_yaw = tf::getYaw(odom_msg->pose.pose.orientation);
}

void yaw_zheng(){
    if(odom_yaw<-0.1){  motion.angular.z=1;  }
    if(odom_yaw>0.1){ motion.angular.z=-1; }
}
void y_0(){
    if(odom_y<-0.01){ motion.linear.y=0.2;  }
    if(odom_y>0.01){  motion.linear.y=-0.2; }
}
void scanCallback(const sensor_msgs::LaserScanConstPtr laser_msg){
        // 将正前方的角度转换为索引  （弧度制）
    int front_index = (int)((0.00- laser_msg->angle_min) / laser_msg->angle_increment);  
        // 将正左方的角度转换为索引  （弧度制）
    int left_index = (int)((1.5707- laser_msg->angle_min) / laser_msg->angle_increment);      
        // 将正右方的角度转换为索引  （弧度制）
    int right_index = (int)((-1.5707- laser_msg->angle_min) / laser_msg->angle_increment);          
        // 将正后方的角度转换为索引  （弧度制）
    int back_index = (int)((3.1416- laser_msg->angle_min) / laser_msg->angle_increment);  
    
    
    // 确保索引在有效范围内  
    //if (front_index >= 0 && front_index < laser_msg->ranges.size())     
        //以小车为主坐标正前方距离
    df = laser_msg->ranges[front_index];
        //以小车为主坐标正左方距离
    dl = laser_msg->ranges[left_index];
        //以小车为主坐标正右方距离
    dr = laser_msg->ranges[right_index];
        //以小车为主坐标正后方距离
    db =laser_msg->ranges[back_index];
    // 检查距离是否有效（即不是无穷大） 
    //if (std::isfinite(df) && std::isfinite(dl) && std::isfinite(dr))


    switch(mission_num){
        case 0:
                        if(odom_x<0.4){                    ////PARAM
                            motion.linear.x=1.5;
                            motion.linear.y=0.0;
                            motion.linear.z=0.0;
                            motion.angular.x=0.0;
                            motion.angular.y=0.0;
                            motion.angular.z=0.0;
                            yaw_zheng();
                            y_0();
                            }
                        else{
                            motion.linear.x=0.0;
                            motion.linear.y=0.0;
                            motion.linear.z=0.0;
                            motion.angular.x=0.0;
                            motion.angular.y=0.0;
                            motion.angular.z=0.0;
                            mission_num=1;
                            }
            break;
        case 1:
                        if(dl>0.40){                    ////PARAM
                            motion.linear.x=0.0;
                            motion.linear.y=1.5;
                            motion.linear.z=0.0;
                            motion.angular.x=0.0;
                            motion.angular.y=0.0;
                            motion.angular.z=0.0;
                            yaw_zheng();
                                if(db<0.41){
                                    motion.linear.x=0.1;
                                    motion.linear.y=1.5;
                                    motion.linear.z=0.0;
                                    motion.angular.x=0.0;
                                    motion.angular.y=0.0;
                                    motion.angular.z=0.0;
                                }   
                                if(db>0.4){
                                    motion.linear.x=-0.1;
                                    motion.linear.y=1.5;
                                    motion.linear.z=0.0;
                                    motion.angular.x=0.0;
                                    motion.angular.y=0.0;
                                    motion.angular.z=0.0;
                                }  
                            }
                        else{
                            motion.linear.x=0.0;
                            motion.linear.y=0.0;
                            motion.linear.z=0.0;
                            motion.angular.x=0.0;
                            motion.angular.y=0.0;
                            motion.angular.z=0.0;
                            mission_num=2;
                            }
            break;
        case 2:
                        if(df>0.57){
                            motion.linear.x=1.5;
                            motion.linear.y=0,0;
                            motion.linear.z=0.0;
                            motion.angular.x=0.0;
                            motion.angular.y=0.0;
                            motion.angular.z=0.0;
                            yaw_zheng();
                            }
                        else{
                            //YOLO

                            //YOLO
                            motion.linear.x=0.0;
                            motion.linear.y=0.0;
                            motion.linear.z=0.0;
                            motion.angular.x=0.0;
                            motion.angular.y=0.0;
                            motion.angular.z=0.0;
                            mission_num=3;
                            }
            break;
        case 3:
                        if(db>0.42){
                            motion.linear.x=-1.5;
                            motion.linear.y=0.0;
                            motion.linear.z=0.0;
                            motion.angular.x=0.0;
                            motion.angular.y=0.0;
                            motion.angular.z=0.0;
                            yaw_zheng();
                            }
                        else{
                            motion.linear.x=0.0;
                            motion.linear.y=0.0;
                            motion.linear.z=0.0;
                            motion.angular.x=0.0;
                            motion.angular.y=0.0;
                            motion.angular.z=0.0;
                            mission_num=4;
                            }
            break;
        case 4:
                        if(odom_y>0.1){             //0
                            motion.linear.x=0.0;
                            motion.linear.y=-1.5;
                            motion.linear.z=0.0;
                            motion.angular.x=0.0;
                            motion.angular.y=0.0;
                            motion.angular.z=0.0;
                            yaw_zheng();
                                if(db<0.41){
                                    motion.linear.x=0.2;
                                    motion.linear.y=-1.5;
                                    motion.linear.z=0.0;
                                    motion.angular.x=0.0;
                                    motion.angular.y=0.0;
                                    motion.angular.z=0.0;
                                }   
                                if(db>0.4){
                                    motion.linear.x=-0.2;
                                    motion.linear.y=-1.5;
                                    motion.linear.z=0.0;
                                    motion.angular.x=0.0;
                                    motion.angular.y=0.0;
                                    motion.angular.z=0.0;
                                }    
                        }
                        else{
                            motion.linear.x=0.0;
                            motion.linear.y=0.0;
                            motion.linear.z=0.0;
                            motion.angular.x=0.0;
                            motion.angular.y=0.0;
                            motion.angular.z=0.0;
                            mission_num=5;
                            }            
            break;
        case 5:                  

                        if(odom_x>-1.67){
                            motion.linear.x=-1.5;
                            motion.linear.y=0.0;
                            motion.linear.z=0.0;
                            motion.angular.x=0.0;
                            motion.angular.y=0.0;
                            motion.angular.z=0.0;
                            yaw_zheng();
                            y_0();
                            back_num--;
                            }
                        else{
                            motion.linear.x=0.0;
                            motion.linear.y=0.0;
                            motion.linear.z=0.0;
                            motion.angular.x=0.0;
                            motion.angular.y=0.0;
                            motion.angular.z=0.0;
                            mission_num=6;
                            }  
            break;
        case 6:
            
            //system("rosrun nav_pub_point nav_pub_1");
            mission_num=7;
            break;
    }
}


int main(int argc,char* argv[]){
    auto start = high_resolution_clock::now();
    ros::init(argc,argv,"laser_direction_print");
    ros::NodeHandle nn;
    ros::Subscriber odom_RX = nn.subscribe("/odom",10,odomCallback);
    ros::Subscriber laser_RX = nn.subscribe("/scan",1000,scanCallback);
    ros::Publisher  control_TX = nn.advertise<geometry_msgs::Twist>("/cmd_vel",1,true);
    MoveBaseClient ac("/move_base",true);
    while(!ac.waitForServer(ros::Duration(5.0))){
        ROS_INFO("waiting for move_base action server");
    }
    while (ros::ok() && mission_num!=7){
        control_TX.publish(motion);
        ros::spinOnce();
    }

      

    /**
     * @brief 声明一个目标点
     * 
     */
    move_base_msgs::MoveBaseGoal goal;
    goal.target_pose.header.frame_id = "map";
    goal.target_pose.header.stamp = ros::Time::now();






    //救援物资位置


    //物资3（位置随即）
    goal.target_pose.pose.position.x = -1.2;
    goal.target_pose.pose.position.y =  1.0;
    goal.target_pose.pose.position.z =  0.0;
    goal.target_pose.pose.orientation.w =1;
    goal.target_pose.pose.orientation.x =0;
    goal.target_pose.pose.orientation.y =0;
    goal.target_pose.pose.orientation.z =0;  
    ac.sendGoal(goal);
    ac.waitForResult();
    if(ac.getState()== actionlib::SimpleClientGoalState::SUCCEEDED){
        //YOLO


        //YOLO
    }
    else{
        ROS_INFO("FAILED");
    }  


    //物资2（位置随即）
    goal.target_pose.pose.position.x = -1.5245239336;
    goal.target_pose.pose.position.y =  1.22600019732;
    goal.target_pose.pose.position.z =  0.0;
    goal.target_pose.pose.orientation.w =0.707107;
    goal.target_pose.pose.orientation.x =0;
    goal.target_pose.pose.orientation.y =0;
    goal.target_pose.pose.orientation.z =0.707107;  
    ac.sendGoal(goal);
    ac.waitForResult();
    if(ac.getState()== actionlib::SimpleClientGoalState::SUCCEEDED){
        //YOLO


        //YOLO
    }
    else{
    }  

    //物资1（位置随即）
    goal.target_pose.pose.position.x = -2.0949111;
    goal.target_pose.pose.position.y =  1.0114200;
    goal.target_pose.pose.position.z =  0.0;
    goal.target_pose.pose.orientation.w =1;
    goal.target_pose.pose.orientation.x =0;
    goal.target_pose.pose.orientation.y =0;
    goal.target_pose.pose.orientation.z =0;  
    // goal.target_pose.pose.orientation.w =0.41;
    // goal.target_pose.pose.orientation.x =0;
    // goal.target_pose.pose.orientation.y =0;
    // goal.target_pose.pose.orientation.z =0.91;  
    ac.sendGoal(goal);
    ac.waitForResult();
    if(ac.getState()== actionlib::SimpleClientGoalState::SUCCEEDED){
        //YOLO


        //YOLO
    }
    else{
        ROS_INFO("FAILED");
    }  












    //救援物资位置


    // 急救包
    goal.target_pose.pose.position.x = -2.490321909699615;
    goal.target_pose.pose.position.y = -1.297248203004;
    goal.target_pose.pose.position.z = 0.0;
    goal.target_pose.pose.orientation.w =-0.709342716993270;
    goal.target_pose.pose.orientation.x =8.580904842586581e-08;
    goal.target_pose.pose.orientation.y =-1.2556726945263934e-07;
    goal.target_pose.pose.orientation.z =0.7048637526845751;
    ac.sendGoal(goal);
    ac.waitForResult();
    if(ac.getState()== actionlib::SimpleClientGoalState::SUCCEEDED){
    }
    else{
        ROS_INFO("FAILED");
    }
    // 台阶后
    goal.target_pose.pose.position.x = -1.62931;
    goal.target_pose.pose.position.y = -0.02750850;
    goal.target_pose.pose.position.z = 0.0;
    goal.target_pose.pose.orientation.w =1.0;
    goal.target_pose.pose.orientation.x =0.0;
    goal.target_pose.pose.orientation.y =0.0;
    goal.target_pose.pose.orientation.z =0.0;
    ac.sendGoal(goal);
    ac.waitForResult();
    if(ac.getState()== actionlib::SimpleClientGoalState::SUCCEEDED){
    }
    else{
        ROS_INFO("FAILED");
    }    

    ros::Rate re(60);  
    ros::Subscriber odom_RX2 = nn.subscribe("/odom",10,odomCallback);         
    while(odom_x<0.0){         //-0.2
    
        ros::spinOnce();
        motion.linear.x=1.5;
        motion.linear.y=0.0;
        motion.linear.z=0.0;
        motion.angular.x=0.0;
        motion.angular.y=0.0;
        motion.angular.z=0.0;
        yaw_zheng();
        if(odom_y>0.016){
        motion.linear.x=1.5;
        motion.linear.y=-0.2;
        motion.linear.z=0.0;
        motion.angular.x=0.0;
        motion.angular.y=0.0;
        motion.angular.z=0.0;            
        }
        if(odom_y<-0.016){
        motion.linear.x=1.5;
        motion.linear.y=0.2;
        motion.linear.z=0.0;
        motion.angular.x=0.0;
        motion.angular.y=0.0;
        motion.angular.z=0.0;            
        }
        control_TX.publish(motion);          
        re.sleep();
    }
        motion.linear.x=0.0;
        motion.linear.y=0.0;
        motion.linear.z=0.0;
        motion.angular.x=0.0;
        motion.angular.y=0.0;
        motion.angular.z=0.0;
        control_TX.publish(motion);





    //复位
    // goal.target_pose.pose.position.x = 0.0;
    // goal.target_pose.pose.position.y = -0.01;
    // goal.target_pose.pose.position.z = 0.0;
    // goal.target_pose.pose.orientation.w = 1.0;
    // goal.target_pose.pose.orientation.x = 0.0;
    // goal.target_pose.pose.orientation.y = 0.0;
    // goal.target_pose.pose.orientation.z = 0.0;
    // ac.sendGoal(goal);
    // ROS_INFO("sended");
    // ac.waitForResult();
    // if(ac.getState()== actionlib::SimpleClientGoalState::SUCCEEDED){
    //     ROS_INFO("taijie_SUCCESS");
    // }
    // else{
    //     ROS_INFO("FAILED");
    // }  
    //对准寻线区
    goal.target_pose.pose.position.x = 0.0;
    goal.target_pose.pose.position.y = -0.01;
    goal.target_pose.pose.position.z = 0.0;
    goal.target_pose.pose.orientation.w = 0.707107;
    goal.target_pose.pose.orientation.x = 0.0;
    goal.target_pose.pose.orientation.y = 0.0;
    goal.target_pose.pose.orientation.z = -0.707107;
    ac.sendGoal(goal);
    ac.waitForResult();
    if(ac.getState()== actionlib::SimpleClientGoalState::SUCCEEDED){
    }
    else{
        ROS_INFO("FAILED");
    }        
    // FOLLOW_LINE
    // system("rosrun follow_line_demo follow_line_demo");
    // FOLLOW_LINE

    //slam路线
    // goal.target_pose.pose.position.x = 2.0880;
    // goal.target_pose.pose.position.y = -0.041143;
    // goal.target_pose.pose.position.z = 0.0;
    // goal.target_pose.pose.orientation.w = 0.99189;
    // goal.target_pose.pose.orientation.x = -0.00013;
    // goal.target_pose.pose.orientation.y = 0.000356;
    // goal.target_pose.pose.orientation.z = -0.127064;
    // ac.sendGoal(goal);
    // ROS_INFO("sended");
    // ac.waitForResult();
    // if(ac.getState()== actionlib::SimpleClientGoalState::SUCCEEDED){
    // }
    // else{
    //     ROS_INFO("FAILED");
    // }      
    auto end = high_resolution_clock::now();
    auto re_time = duration_cast<std::chrono::milliseconds>(end-start);
    // 设置亮绿色文本  
    std::cout << "\033[92m"; // 开始亮绿色文本  
    std::cout << "总耗时 " << re_time.count()/1000.0 << "s" << std::endl;  
    std::cout << "\033[0m"; // 重置到默认颜色  



    //测试，回到原点（比赛勿开）
    goal.target_pose.pose.position.x = 0;
    goal.target_pose.pose.position.y =0;
    goal.target_pose.pose.position.z = 0.0;
    goal.target_pose.pose.orientation.w = 1;
    goal.target_pose.pose.orientation.x = 0;
    goal.target_pose.pose.orientation.y = 0;
    goal.target_pose.pose.orientation.z = 0;
    ac.sendGoal(goal);
    ac.waitForResult();
    if(ac.getState()== actionlib::SimpleClientGoalState::SUCCEEDED){
    }
    else{
        ROS_INFO("FAILED");
    }     
    ros::shutdown();
    system("rosrun laser_direction_control_demo laser_direction_control_demo");
    return 0;
}