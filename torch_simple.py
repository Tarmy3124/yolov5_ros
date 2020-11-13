#!/usr/bin/env python

import torch
from PIL import Image, ImageDraw 
import numpy as np
from models.experimental import attempt_load



import rospy
import cv2
import sys
from std_msgs.msg import String
from sensor_msgs.msg import Image as Img
from cv_bridge import CvBridge, CvBridgeError

class image_converter:

    def callback(self,data):
        try:
          img = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
          print(e)

        # Model
        model = attempt_load('/home/lunabot87/robot_ws/src/yolov5/yolov5_ros/chan_hole_best.pt', map_location='cuda')
        model = model.autoshape()  # for autoshaping of PIL/cv2/np inputs and NMS

        # Inference
        prediction = model(img)  # includes NMS
        box_pose_list = []

        for i, pred in enumerate(prediction):
            img = Image.fromarray(img.astype(np.uint8)) if isinstance(img, np.ndarray) else img  # from np
            if pred is not None:
                for c in pred[:, -1].unique():
                    n = (pred[:, -1] == c).sum()  # detections per class
                for *box, conf, cls in pred:  # xyxy, confidence, class
                    label = model.names[int(cls)] if hasattr(model, 'names') else 'class_%g' % cls
                    # str += '%s %.2f, ' % (label, conf)  # label
                    ImageDraw.Draw(img).rectangle(box, width=3, outline ="red")  # plot
                    open_cv_image = np.array(img) 
                    # Convert RGB to BGR 
                    open_cv_image = open_cv_image[:, :, ::-1].copy()
                    
                    temp = []
                   # print("box : {0}".format(box))
                    for x in box:
                        temp.append(x.tolist())
                    box_pose_list.append(temp)

                print(box_pose_list)
                print("end")
            # img.save('results%g.jpg' % i)  # save

        try:
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(open_cv_image, "bgr8"))
        except CvBridgeError as e:
            print(e)


    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("camera1/usb_cam1_1/image_raw",Img,self.callback)
        self.image_pub = rospy.Publisher("image_topic_2",Img)


def main(args):
    ic = image_converter()
    rospy.init_node('image_converter', anonymous=True)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)