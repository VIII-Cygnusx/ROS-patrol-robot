/**
 * @file cv_get.cpp
 * @author Cygnusx (Cygnusx@domain.com)
 * @brief 订阅摄像头话题并转为Opencv格式进行处理的包(学习代码)
 * @version 0.1
 * @date 2024-03-08
 * 
 * @copyright Copyright (c) 2024
 * 
 */
#include "ros/ros.h"
#include "sensor_msgs/Image.h"
#include "opencv4/opencv2/opencv.hpp"
/**
 * @brief cv_bridge 是一个ROS包，它提供了一个桥梁，
 *        使得开发者可以在ROS节点中方便地
 *        使用OpenCV库来处理图像
*/
#include "cv_bridge/cv_bridge.h"

#include "sys/socket.h"
#include <arpa/inet.h> // for inet_addr, htons
#include <unistd.h> // for close()
#include <cstring> // for memcpy


using namespace ros;
using namespace cv;
using namespace sensor_msgs;
using namespace cv_bridge;


/**
 * @brief 是一个回调函数
 * @param msg 是一个智能指针它指向一个常量 sensor_msgs::Image 
 *            这个智能指针确保在回调函数中使用的消息是有效的，
 *            并且在消息不再需要时被正确地删除
*/
void image_Callback(const sensor_msgs::ImageConstPtr& msg);

int main(int argc,char *argv[]){
    ros::init(argc,argv,"cv_get");
    ros::NodeHandle nh;
    /**
     * @brief 来创建一个订阅者（Subscriber），用于订阅"/cam"的消息
     * @param get_cv            订阅者
     * @param nh                用于与ROS系统交互,创建订阅者
     * @param cam               订阅的话题名
     * @param 1000              队列缓存大小
     * @param image_Callback    是一个回调函数        
    */
    ros::Subscriber get_cv = nh.subscribe("/cam",1000,image_Callback);



    /**
     * @brief 它用于启动ROS的事件循环。当你编写一个ROS节点（node）时，
     *        该节点通常会订阅（subscribe）一些话题（topic）或提供服务（service），
     *        并等待来自其他节点的消息或服务调用。ros::spin()就是使你的节点保持运行状态，
     *        并处理这些消息和调用的函数。
     * @warning 此函数本身并不提供直接设置频率的参数
    */
    ros::spin();
    /**
     * @brief 确保关闭时关闭
    */
    nh.shutdown();
    return 0;
}



void image_Callback(const sensor_msgs::ImageConstPtr& msg){
    /**
     *  @brief try{}catch(){}       用于异常处理的机制
     *  @note try 块用于包含可能引发异常的代码，而 catch 块则用于捕获并处理这些异常
    */
    try{
        /**
         * @param cv_ptr 是一个智能指针类型指向一个cv_bridge::CvImage,
         *               这个智能指针确保在回调函数中使用的消息是有效的，
         *               并且在消息不再需要时被正确地删除
         *               里面封装了一个cv::Mat对象（OpenCV中的图像表示
         *               以及关于该图像的一些元信息（如时间戳、图像编码格式等）。
        */
        cv_bridge::CvImageConstPtr cv_ptr;

        /**
         * @brief 用于将 ROS 的 sensor_msgs::Image 消息转换为一个共享引用的 OpenCV cv::Mat 对象
         * @note cv_bridge::toCvShare与cv_bridge::toCvCopy不同，
         *       cv_bridge::toCvShare 不会复制图像数据，而是创建一个指向原始 ROS 图像数据的引用。
         *       这意味着转换后的 OpenCV 图像和原始的 ROS 图像消息将共享同一块内存。
         *       (使用 cv_bridge::toCvShare 可以提高性能，因为它避免了不必要的数据复制)
         * 
         * @param sensor_msgs::image_encodings::BGR8 表示用于指示图像数据的格式
         *                                           表明该 Image 消息中的 data 字段包含以 BGR 顺序排列的 8 位颜色值
        */
        cv_ptr = cv_bridge::toCvShare(msg,sensor_msgs::image_encodings::BGR8);
        /**
         * @brief 创立一个名为CV_GET的窗口显示图片
        */
        cv::imshow("CV_GET",cv_ptr->image);
        if(cv::waitKey(1)>=0){
            return ;
        }

        cv:imwrite("/home/cygnusx/LINUX/ROS/my_gazebo_sim_pro/src/ONNX/run.png",cv_ptr->image);
        system("/home/cygnusx/miniconda3/envs/AI/bin/python /home/cygnusx/LINUX/ROS/my_gazebo_sim_pro/src/ONNX/ONNX_detect.py");












    /**
     * @brief  定义一个异常类cv_bridge::Exception
     * @param  e 是一个异常类
     * @note 如果发生错误（例如，不支持的图像编码），则可能会抛出cv_bridge::Exception异常
     */   
    }catch(cv_bridge::Exception& e){
        /**
         * @param msg->encoding.c_str() 是一个字符串，用于标识图像数据的编码格式    
         *                              例如，它可能是sensor_msgs::image_encodings::BGR8
         *                              c_str()是C++标准库中std::string类
        */
        ROS_ERROR("no image from '%s' to 'bgr8'.",msg->encoding.c_str());
    }
}