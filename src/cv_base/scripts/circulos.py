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
      #img_gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
            
      #h,w,c = rgb_image.shape
      #dsize = (int(w*0.250), int(h*0.250))
      # resize image
      #imgCir = cv2.resize(rgb_image, dsize)
               
      '''src = cv2.medianBlur(rgb_image, 5)
      img_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
      
            
      circles = cv2.HoughCircles(img_gray, cv2.HOUGH_GRADIENT, 1, 20,
                            param1=50, param2=80, minRadius=0, maxRadius=0)

      circles = np.uint16(np.around(circles))
      imgCir = rgb_image
      for i in circles[0,:]:
        # dibujar circulo 
        cv2.circle(imgCir, (i[0], i[1]), i[2], (0,255,0), 2)
        # dibujar centro
        cv2.circle(imgCir, (i[0], i[1]), 2, (0,0,255), 3)
      
      final_image =  imgCir  '''
      #image = cv2.imread('1.jpg')
      gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
      thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

      # Find circles with HoughCircles
      circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, minDist=20, param1=50, param2=80, minRadius=0)
      # Find circles with HoughCircles
      #circles = cv2.HoughCircles(thresh, cv2.HOUGH_GRADIENT, 1, minDist=150, param1=200, param2=18, minRadius=20)


      # Draw circles
      if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x,y,r) in circles:
            cv2.circle(rgb_image, (x,y), r, (36,255,12), 3)
      return rgb_image

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
