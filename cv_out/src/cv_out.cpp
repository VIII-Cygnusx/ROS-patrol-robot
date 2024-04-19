/**
 * @file cv_out.cpp
 * @author Cygnusx (Cygnusx@domain.com)
 * @brief 打开usb摄像头并以/computer_cam话题名发布(测试版)
 * @version 0.1
 * @date 2024-03-08
 * 
 * @copyright Copyright (c) 2024
 * 
 */
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <opencv4/opencv2/opencv.hpp>
#include <cv_bridge/cv_bridge.h>
#include <iostream>
using namespace std;
using namespace ros;
using namespace cv;
using namespace sensor_msgs;
using namespace cv_bridge;


sensor_msgs::Image CVimg2ROSimg(const cv::Mat& cv_image){
    sensor_msgs::Image ros_img;
    ros_img.header.stamp = ros::Time::now();
    ros_img.header.frame_id = "camera";
    ros_img.encoding = sensor_msgs::image_encodings::BGR8;
    ros_img.width = cv_image.cols;
    ros_img.height = cv_image.rows;
    ros_img.is_bigendian = false;
    ros_img.step = cv_image.step;
    size_t required_size = cv_image.total()*cv_image.elemSize();
    ros_img.data.resize(required_size);
    if(ros_img.data.size()<required_size){
        ROS_ERROR("something warn");
        return ros_img;
    }
    std::memcpy(ros_img.data.data(),cv_image.data,ros_img.data.size());
    return ros_img;
}

int main(int argc,char* argv[]){
    ros::init(argc,argv,"cv_out");
    ros::NodeHandle nh;
    ros::Publisher out_cv = nh.advertise<sensor_msgs::Image>("/computer_cam",1000,true);
    cv::VideoCapture cap;
    cap.open(0);
    if (!cap.isOpened()) {
        ROS_ERROR("no signals!!");
    }
    ros::Rate r(30);        //30帧
    while(ros::ok()){
        cv::Mat frame;
        cap >> frame;
        if(!frame.empty()){
            cv::Size R(320,240);
            cv::resize(frame,frame,R,0,0,cv::INTER_LINEAR);
            sensor_msgs::Image ros_get_img=CVimg2ROSimg(frame);
            cv::imshow("Vedio",frame);
            out_cv.publish(ros_get_img);
        }
        if(cv::waitKey(1)>=0){
            break;
        }
        r.sleep();
    }
    nh.shutdown();
    cv::destroyAllWindows();
    return 0;
}