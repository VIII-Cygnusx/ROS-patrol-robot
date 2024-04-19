#################################################################################
#    本格式为pt格式转onnx，开放的机器学习格式可后续转多模型格式以适配不同gpu或npu
#    适配张量:[1,3,640,640]     
#    输出张量:[1,25200,85]   85中前面5个分别为     后面80个为class类   
#    本模型推理适配为yolov5.7训练出来的模型,只能作为图形识别设置
#    纯cpu运算
#    把providers=['CPUExecutionProvider'] 改为providers=['CUDAExecutionProvider']使用cuda运算
#    去https://netron.app 查看当前模型结构，本代码专门为此结构编写没有移植性
#    大便程序狗屎优化，帧数感人
#    本模型可在yolov5官网下载直接自带人脸识别，须5.7-5.x版本的模型
#    Version:1.0.7      Data:2023/12/1
#################################################################################
import onnx
import onnxruntime as ort   
import numpy as np
import cv2              
import pandas as pd     #整理张量并打包到csv做可视化
from playsound import playsound











CLASSES = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90]  #类别





def mp3_play(file_path):
    playsound(file_path)






class yolo5ONNX(object):
    """获取输出形状"""
    def get_output_shape(self):
        output_shape = []
        for node in self.onnx_session.get_outputs():
            output_shape.append(node.name)
            print(node.name)
        return output_shape
    """添加一个标签"""  
    def get_input_tensor(self,image):
        input_feed={}
        input_feed[self.onnx_model.graph.input[0].name] = image
        return input_feed

    
    def xywh2xyxy(self,x):
        # [x, y, w, h] to [x1, y1, x2, y2]
        y = np.copy(x)
        y[..., 0] = x[..., 0] - x[..., 2] / 2
        y[..., 1] = x[..., 1] - x[..., 3] / 2
        y[..., 2] = x[..., 0] + x[..., 2] / 2
        y[..., 3] = x[..., 1] + x[..., 3] / 2
        return y
    
    """框选阈值#<---------------------------------pusule
    ┌------------┐
    |    ┌-------┼----┐
    |    |///////|    |
    |    |///////|    |     thresh与重叠部分的阈值，只考虑了其中情况
    └----┼-------┘    |                                         
         └------------┘
    """                
    def nms(self,dets, thresh):
        # dets:x1 y1 x2 y2 score class
        # x[:,n]就是取所有集合的第n个数据
        x1 = dets[..., 0]
        y1 = dets[..., 1]
        x2 = dets[..., 2]
        y2 = dets[..., 3]
        scores = dets[..., 4]
        # -------------------------------------------------------
        #   计算框的面积
        #	置信度从大到小排序
        # -------------------------------------------------------
        areas = (y2 - y1 + 1) * (x2 - x1 + 1)
        
        print("每行分数",scores)
        keep = []
        index = scores.argsort()[::-1]  # np.argsort()对某维度从小到大排序
        print("从小到大排序",index)
        print("测试",x1[index[1:]])
        # [::-1] 从最后一个元素到第一个元素复制一遍。倒序从而从大到小排序

        while index.size > 0:
            i = index[0]
            keep.append(i)
            # -------------------------------------------------------
            #   计算相交面积
            #	1.相交
            #	2.不相交
            # -------------------------------------------------------
            x11 = np.maximum(x1[i], x1[index[1:]]) 
            y11 = np.maximum(y1[i], y1[index[1:]])
            x22 = np.minimum(x2[i], x2[index[1:]])
            y22 = np.minimum(y2[i], y2[index[1:]])

            w = np.maximum(0, x22 - x11 + 1)
            h = np.maximum(0, y22 - y11 + 1)

            overlaps = w * h    #相交面积
            # -------------------------------------------------------
            #   计算该框与其它框的IOU，去除掉重复的框，即IOU值大的框
            #	IOU小于thresh的框保留下来
            # -------------------------------------------------------
            ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)  #相交面积/所有区域面积
            idx = np.where(ious <= thresh)[0]
            index = index[idx + 1]
        return keep
    """初始化"""
    def __init__(self,onnx_path) -> None:
        self.onnx_model=onnx.load(onnx_path)
        try:
            onnx.checker.check_model(self.onnx_model)
        except Exception:       #处理异常
            print("incorrect")
        else:
            print("=========================")
            print("model_correct!!")
            print("model_path:",onnx_path)
            print("=========================")
        pass
        self.options = ort.SessionOptions()   # 创建一个ort.SessionOptions对象，用于配置ONNX模型运行选项
        self.options.enable_profiling = True  # 开启性能分析
        #创建一个ONNX模型的推理会话ort.InferenceSession来加载和运行ONNX模型
        self.onnx_session = ort.InferenceSession(path_or_bytes=onnx_path,               #参数指定了ONNX模型文件的路径
                                                 sess_options=self.options,             #用于配置推理会话的选项
                                                 providers=['CUDAExecutionProvider'])    #指定了使用的执行提供程序
        print("USE",self.onnx_session.get_providers())
        print("输入张量",self.onnx_model.graph.input[0].type.tensor_type.shape.dim)  #打印输入张量的形状
        self.output_shape=self.get_output_shape()                         #输出形状
        print("=========================")


    """推理"""
    def inference(self,img):
        img_width=self.onnx_model.graph.input[0].type.tensor_type.shape.dim[2].dim_value        #图形宽
        img_height=self.onnx_model.graph.input[0].type.tensor_type.shape.dim[3].dim_value       #图形长
        resize_pic=cv2.resize(img,(img_width,img_height))                              #调整大小
        change_rgb_channel=resize_pic[:,:,::-1].transpose(2,0,1)                       #转换通道
        change_img_type=change_rgb_channel.astype(np.float16)                          #转换类型onnx模型的类型是type: float32
        change_img_type/=255.0                                                         #归一化
        change_img_tensor = np.expand_dims(change_img_type, axis=0)                    #变成模型输入张量格式[3, 640, 640]扩展为[1, 3, 640, 640]
        self.input_tensor =self.get_input_tensor(change_img_tensor)                    #这一步是加上标签          
        result=self.onnx_session.run(None,self.input_tensor)[0]                        #开始推理           
        return result,resize_pic
    
    def filter_box(self,org_box,conf_thres,iou_thres):
        #把张量处理成类别
        org_box=np.squeeze(org_box)                                               #去掉单一的维度只有1的维度
        #print("去掉无用维度的张量形状",org_box.shape)                              #去掉单一的维度的张量(25200列, 9行)
        conf=org_box[...,4]>conf_thres                                            #取出置信度,所有维度的第5个元素（索引为4）    #置信度就是socre
        box=org_box[conf==True]                                                   #过滤出符合的元素整合为新的符合要求的张量                         
        #print("符合的置信度张量形状",box.shape)                                    
        temp = pd.DataFrame(box)                                                  #将张量转换为DataFrame
        #temp.to_csv('tensor_look/符合的置信度张量形状.csv', index=False)                       #将DataFrame保存为csv文件

        cut_box=box[...,5:]                                                       #切出25200行后面从6到9的元素组成一个新张量
        #print("切出来的形状",cut_box.shape)                                        
        temp = pd.DataFrame(cut_box)                                                  #将张量转换为DataFrame
        #temp.to_csv('tensor_look/切出来的形状.csv', index=False)

        max_cell=[]                                                               #每行最大的数     
        #print("=========================")  
        for i in range(len(cut_box)):                                             #长
            max_cell.append(np.argmax(cut_box[i]))                                #找到每行最大值的索引值
            #print(i,"行最大值的索引值为",max_cell[i])                              #
        temp = pd.DataFrame(max_cell)                                             #将张量转换为DataFrame
        #temp.to_csv('tensor_look/每行最大值的索引.csv', index=False)    
        #print("=========================")
        


        det_cls = list(set(max_cell))                                             # 去重，找出图中都有哪些类别
        print("总识别数",det_cls)


        outputs=[]
        #分别处理识别到的类别
        for i in range(len(det_cls)):                                             
            now_class=det_cls[i]
            now_cls_box=[]
            now_out_box=[]
            #处理识别到的数量
            for j in range(len(max_cell)):
                if max_cell[j]==now_class:
                    box[j][5]=now_class                                           #直接把该张量class的第一位覆盖为识别到的索引值
                    now_cls_box.append(box[j][:6])                                #提取张量整行从1到6的所有元素    0 1 2 3 4 5 分别是 x y w h score class                      
            now_cls_box=np.array(now_cls_box)                                         #转换为numpy数组  0 1 2 3 4 5 分别是 x y w h score class
            now_cls_box=self.xywh2xyxy(now_cls_box)                                   #坐标转换
            now_out_box=self.nms(now_cls_box,iou_thres)                               #边界框处理
            for k in now_out_box:
                outputs.append(now_cls_box[k])
        outputs=np.array(outputs)
        print("输出结果",outputs)
        return outputs
    
    """根据处理出来的坐标轴画图"""
    def draw(self,image,box_data):
        boxs=box_data[...,:4].astype(np.int32)                   # x1 y1 x2 y2
        scores=box_data[...,4]                                  #分数
        class_names=box_data[...,5].astype(np.int32)             #类别
        for box,score,class_name in zip(boxs,scores,class_names):       #多处理
            x1,y1,x2,y2=box                                     
            #print('类: {}, 分数: {}'.format(CLASSES[class_name], score))
            #print('box: x1: {}, x2: {}, y1: {}, y2: {}'.format(x1, x2, y1, y2))
            cv2.rectangle(image,(x1,y1),(x2,y2),(0,255,0),1)                        #画框
            cv2.putText(img=image, 
                        text='{0} {1:.2f}'.format(CLASSES[class_name], score),    #写字
                        org=(x1, y1),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.6, 
                        color=(0, 0, 255), 
                        thickness=2
                        )
        return image
    

    """视频实时检测"""
    def video_cap(self):
        cap = cv2.VideoCapture(0)               #直连摄像头
        while(True):
            ret,frame=cap.read()
            pic=frame
            result,reshape_pic=self.inference(pic)
            #print("结果张量",result.shape)
            out_box=self.filter_box(result,0.6,0.6)
            if(len(out_box)==0):
                print("没有检测到目标")
                final_img=reshape_pic
            else:
                final_img=self.draw(reshape_pic,out_box)
                mp3_play("/home/cygnusx/LINUX/ROS/my_gazebo_sim_pro/src/ONNX/speck.wav") 
            cv2.imshow("final_img",final_img)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()


    """图片检测"""
    def image_cap(self,image_path):
        cap=cv2.imread(image_path)
        result,reshape_pic=self.inference(cap)
        print("结果张量",result.shape)
        out_box=self.filter_box(result,0.5,0.6)   #最佳置信度边框损失没有加，0.1的边界框重合损失，分类损失内置就是寻找最大索引
        if(len(out_box)==0):
            print("没有检测到目标")
            final_img=reshape_pic
        else:
            final_img=self.draw(reshape_pic,out_box)
            cv2.imwrite('/home/cygnusx/LINUX/ROS/my_gazebo_sim_pro/src/ONNX/run_.png',final_img)
            # cv2.imshow("final_img",final_img)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
# def callback(imgs):
#     imc = CvBridge()
#     img = imc.imgmsg_to_cv2(imgs,"rgb8")
#     cap=cv2.imshow("onnx",img)





onnx_path='/home/cygnusx/LINUX/ROS/my_gazebo_sim_pro/src/ONNX/yolov5m.onnx'
model=yolo5ONNX(onnx_path)
# rospy.init_node("ONNX",anonymous=True)
# rospy.Subscriber("/cam",Image,callback=callback)
# rospy.spin()
model.video_cap()











