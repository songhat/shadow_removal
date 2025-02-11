#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from shodow_removal.srv import ShadowRemoval
from cv_bridge import CvBridge, CvBridgeError
import cv2

def shadow_removal_client(image_path):
    rospy.wait_for_service('shadow_removal')
    try:
        shadow_removal = rospy.ServiceProxy('shadow_removal', ShadowRemoval)
        
        bridge = CvBridge()
        
        # 读取本地图片
        cv_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        
        # 将OpenCV格式的图片转换为ROS图像消息
        image_message = bridge.cv2_to_imgmsg(cv_image, "bgr8")
        
        # 调用服务
        resp = shadow_removal(image_message)
        
        #显示处理后的图片
        processed_image = bridge.imgmsg_to_cv2(resp.output_image, "bgr8")
        output_path = 'img/shadow_removed_image.jpg'
        rospy.loginfo(f"处理后的图片已保存到 {output_path}")
        cv2.imwrite(output_path, processed_image)
        

    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)

if __name__ == "__main__":
    rospy.init_node('shadow_removal_client', anonymous=True)  # 初始化 ROS 节点
    shadow_removal_client('/home1/rhs/code_workspace/ST-CGAN/imgs/img3.jpg')