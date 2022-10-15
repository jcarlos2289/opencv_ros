#!/usr/bin/env python
#from __future__ import print_function

import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np

class image_mod:
                               
  def modifyImg(self, rgb_image):  #: np.NDArray
      img_gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
      img_blur = cv2.medianBlur(img_gray, 7)
      final_image = cv2.cvtColor(img_blur, cv2.COLOR_GRAY2RGB)           
      return final_image

  def __init__(self):
    self.image_pub = rospy.Publisher("mod_image",Image, queue_size=1)
    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("/cv_camera/image_raw",Image,self.callback)

  def callback(self,data):
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)
    

    mod_image = self.modifyImg(cv_image)  
    '''cv2.imshow("Image window", cv_image)

    if cv2.waitKey(20) & 0xFF == ord('s'):
      cv2.imwrite('mod_image.jpeg', mod_image)

    cv2.waitKey(3)'''

    try:
      self.image_pub.publish(self.bridge.cv2_to_imgmsg(mod_image, "bgr8"))
    except CvBridgeError as e:
      print(e)

def main(args):
  rospy.init_node('image_mod', anonymous=True)
  ic = image_mod()
  try:
    rospy.spin()
  except KeyboardInterrupt:
    print("Shutting down")
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
