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
            
      #detect
      gray_frame = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
      corners, ids, rejected_corners = cv2.aruco.detectMarkers(gray_frame, self.aruco_dictionary, parameters=self.parameters)
      
      # Draw detected markers:
      final_image = cv2.aruco.drawDetectedMarkers(image=rgb_image, corners=corners, ids=ids, borderColor=(0, 255, 0))

      # Draw rejected markers:
      #final_image = cv2.aruco.drawDetectedMarkers(image=frame, corners=rejected_corners, borderColor=(0, 0, 255))
      
      return final_image

  def __init__(self):
    self.image_pub = rospy.Publisher("mod_image",Image, queue_size=1)
    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("/cv_camera/image_raw",Image,self.callback)
    self.aruco_dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_7X7_250)

    # We create the parameters object:
    self.parameters = cv2.aruco.DetectorParameters_create()

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
