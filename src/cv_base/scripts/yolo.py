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
        (H, W) = rgb_image.shape[:2]

        # Get the output layer names:
        layer_names = self.net.getLayerNames()
        layer_names = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

        # Create the blob with a size of (416, 416), swap red and blue channels
        # and also a scale factor of 1/255 = 0,003921568627451:
        blob = cv2.dnn.blobFromImage(rgb_image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
           
        # Feed the input blob to the network, perform inference and get the output:
        self.net.setInput(blob)
        layerOutputs = self.net.forward(layer_names)
        
        # Initialization:
        boxes = []
        confidences = []
        class_ids = []
        
        
        # loop over each of the layer outputs
        for output in layerOutputs:
            # loop over each of the detections
            for detection in output:
                # Get class ID and confidence of the current detection:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                # Filter out weak predictions:
                if confidence > 0.25:
                    # Scale the bounding box coordinates (center, width, height) using the dimensions of the original image:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    # Calculate the top-left corner of the bounding box:
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    # Update the information we have for each detection:
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # We can apply non-maxima suppression (eliminate weak and overlapping bounding boxes):
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)
        
        # Show the results (if any object is detected after non-maxima suppression):
        if len(indices) > 0:
            for i in indices.flatten():
                # Extract the (previously recalculated) bounding box coordinates:
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                # Draw label and confidence:
                cv2.rectangle(rgb_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                label = "{}: {:.4f}".format(self.class_names[class_ids[i]], confidences[i])
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
                y = max(y, labelSize[1])
                cv2.rectangle(rgb_image, (x, y - labelSize[1]), (x + labelSize[0], y + 0), (0, 255, 0), cv2.FILLED)
                cv2.putText(rgb_image, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        return rgb_image

  def __init__(self):
    self.image_pub = rospy.Publisher("mod_image",Image, queue_size=1)
    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("/cv_camera/image_raw",Image,self.callback)
       
          
    # load the COCO class labels:
    self.class_names = open("/home/jet3/Desarrollo/cv_ros/yolo/coco.names").read().strip().split("\n")
    
    
    # Load the serialized caffe model from disk:
    self.net = cv2.dnn.readNetFromDarknet("/home/jet3/Desarrollo/cv_ros/yolo/yolov3.cfg", "/home/jet3/Desarrollo/cv_ros/yolo/yolov3.weights")
    
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
