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
       blob = cv2.dnn.blobFromImage(rgb_image, 1, (224, 224), (104, 117, 123))
      
       # Feed the input blob to the network, perform inference and ghe the output:
       self.net.setInput(blob)
       preds = self.net.forward()
    
       # Get the 10 indexes with the highest probability (in descending order)
       # This way, the index with the highest prob (top prediction) will be the first:
       indexes = np.argsort(preds[0])[::-1][:10]
      
       # We draw on the image the class and probability associated with the top prediction:
       text = "label: {}\nprobability: {:.2f}%".format(self.classes[indexes[0]], preds[0][indexes[0]] * 100)
       y0, dy = 30, 30
       for i, line in enumerate(text.split('\n')):
          y = y0 + i * dy
          cv2.putText(rgb_image, line, (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
      
           
       return rgb_image

  def __init__(self):
    self.image_pub = rospy.Publisher("mod_image",Image, queue_size=1)
    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("/cv_camera/image_raw",Image,self.callback)
    # Load the names of the classes:
    self.rows = open('/home/jet3/Desarrollo/cv_ros/resnet_Live/synset_words.txt').read().strip().split('\n')
    self.classes = [r[r.find(' ') + 1:].split(',')[0] for r in self.rows]
    # Load the serialized caffe model from disk:
    self.net = cv2.dnn.readNetFromCaffe("/home/jet3/Desarrollo/cv_ros/resnet_Live/ResNet-50-deploy.prototxt", "/home/jet3/Desarrollo/cv_ros/resnet_Live/ResNet-50-model.caffemodel")


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
