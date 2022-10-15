#!/usr/bin/env python
from __future__ import print_function

import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import numpy as np

class image_converter:
                                #: np.NDArray
  def cartoonize(self, rgb_image, num_pyr_downs=2,  num_bilaterals=7):
      # STEP 1 -- Apply a bilateral filter to reduce the color palette of
      # the image.
      
      
      
      downsampled_img = rgb_image
      for _ in range(num_pyr_downs):
          downsampled_img = cv2.pyrDown(downsampled_img)

      for _ in range(num_bilaterals):
          filterd_small_img = cv2.bilateralFilter(downsampled_img, 9, 9, 7)

      filtered_normal_img = filterd_small_img
      for _ in range(num_pyr_downs):
          filtered_normal_img = cv2.pyrUp(filtered_normal_img)

      # make sure resulting image has the same dims as original
      if filtered_normal_img.shape != rgb_image.shape:
          filtered_normal_img = cv2.resize(
              filtered_normal_img, rgb_image.shape[:2])

      # STEP 2 -- Convert the original color image into grayscale.
      img_gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
      # STEP 3 -- Apply a median blur to reduce image noise.
      img_blur = cv2.medianBlur(img_gray, 7)

      # STEP 4 -- Use adaptive thresholding to detect and emphasize the edges
      # in an edge mask.
      gray_edges = cv2.adaptiveThreshold(img_blur, 255,
                                         cv2.ADAPTIVE_THRESH_MEAN_C,
                                         cv2.THRESH_BINARY, 9, 2)
      # STEP 5 -- Combine the color image from step 1 with the edge mask
      # from step 4.
      rgb_edges = cv2.cvtColor(gray_edges, cv2.COLOR_GRAY2RGB)
      cartoon_image = cv2.bitwise_and(filtered_normal_img, rgb_edges)
      
          
      return cartoon_image

  def __init__(self):
    self.image_pub = rospy.Publisher("image_topic_2",Image, queue_size=1)

    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("/cv_camera/image_raw",Image,self.callback)

  def callback(self,data):
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)


    #(rows,cols,channels) = cv_image.shape
    #if cols > 60 and rows > 60 :
    #  cv2.circle(cv_image, (50,50), 10, 255)

    #cv_image = cv2.medianBlur(cv_image, 7)    #:np.ndarray

    cv_image = self.cartoonize(cv_image)  #np.asarray(cv_image)

    cv2.imshow("Image window", cv_image)

    if cv2.waitKey(20) & 0xFF == ord('s'):
      cv2.imwrite('cartoon_image.jpeg', cv_image)

    cv2.waitKey(3)

    try:
      self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
    except CvBridgeError as e:
      print(e)

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
