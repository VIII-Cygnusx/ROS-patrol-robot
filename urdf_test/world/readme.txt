若更改了urdf文件夹内xacro文件参数或者改变了gazebo模型之类的
请务必在launch文件中找到 <!-- 载入一个gazebo的空模拟世界 --> 标签
然后把 <arg name="world_name" value="$(find urdf_test)/xxx.world"/> 注释掉
然后再启动launch重新配置gazebo界面中的设置，再保存世界
然后把 <arg name="world_name" value="$(find urdf_test)/xxx.world"/> 取消注释掉1
再次运行
