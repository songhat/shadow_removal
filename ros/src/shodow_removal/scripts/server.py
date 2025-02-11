#!/usr/bin/env python

import numpy as np
import rospy
from sensor_msgs.msg import Image
from shodow_removal.srv import ShadowRemoval, ShadowRemovalResponse
from cv_bridge import CvBridge, CvBridgeError
import cv2
from ST_CGAN.inference import predict
import PIL.Image as PILImage
def handle_shadow_removal(req):
    bridge = CvBridge()
    try:
        # 将ROS图像消息转换为OpenCV格式
        cv_image = bridge.imgmsg_to_cv2(req.input_image, "bgr8")
        checkpoints = [
            '/home1/rhs/code_workspace/ST-CGAN/checkpoints0/ST-CGAN_G1_250.pth',
            '/home1/rhs/code_workspace/ST-CGAN/checkpoints0/ST-CGAN_G2_250.pth'
        ]
        rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        pil_image = PILImage.fromarray(rgb_image)
        shodow_removed_image = predict(pil_image, checkpoints, image_size=256)

        numpy_image = np.array(shodow_removed_image)

        shodow_removed_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)

        # 将处理后的图像转换回ROS图像消息
        modified_image_message = bridge.cv2_to_imgmsg(shodow_removed_image, "bgr8")
        rospy.loginfo("去除阴影完毕，返回图片")
        return ShadowRemovalResponse(modified_image_message)
    except Exception as e:
        import traceback
        traceback.print_exc()

def shadow_removal_server():
    rospy.init_node('shadow_removal_server')
    s = rospy.Service('shadow_removal', ShadowRemoval, handle_shadow_removal)
    rospy.loginfo("Ready to remove shadows from images.")
    rospy.spin()

if __name__ == "__main__":
    shadow_removal_server()