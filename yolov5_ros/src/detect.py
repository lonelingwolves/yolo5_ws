#!/usr/bin/env python3

import rospy
import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from cv_bridge import CvBridge
from pathlib import Path
import os
import sys
from rostopic import get_topic_type

from sensor_msgs.msg import Image
from detection_msgs.msg import BoundingBox, BoundingBoxes


# add yolov5 submodule to path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0] / "yolov5"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative path

# import from yolov5 submodules
from models.common import DetectMultiBackend
from utils.general import (
    check_img_size,
    check_requirements,
    non_max_suppression,
    scale_coords
)
from utils.plots import Annotator, colors
from utils.torch_utils import select_device
from utils.augmentations import letterbox


@torch.no_grad()
class Yolov5Detector:
    def __init__(self):
        self.conf_thres = rospy.get_param("~confidence_threshold")
        self.iou_thres = rospy.get_param("~iou_threshold")
        self.agnostic_nms = rospy.get_param("~agnostic_nms")
        self.max_det = rospy.get_param("~maximum_detections")
        self.classes = rospy.get_param("~classes", None)
        self.line_thickness = rospy.get_param("~line_thickness")
        self.view_image = rospy.get_param("~view_image")
        
        # Initialize weights 
        weights = rospy.get_param("~weights")
        
        # Initialize model
        self.device = select_device(str(rospy.get_param("~device","")))
        self.model = DetectMultiBackend(weights, device=self.device, dnn=rospy.get_param("~dnn"), data=rospy.get_param("~data"))
        self.stride, self.names, self.pt, self.jit, self.onnx, self.engine = (
            self.model.stride,
            self.model.names,
            self.model.pt,
            self.model.jit,
            self.model.onnx,
            self.model.engine,
        )

        # Setting inference size
        self.img_size = [rospy.get_param("~inference_size_w", 640), rospy.get_param("~inference_size_h",480)]
        self.img_size = check_img_size(self.img_size, s=self.stride)

        # Half precision (optional)
        self.half = rospy.get_param("~half", False)
        self.half &= (
            self.pt or self.jit or self.onnx or self.engine
        ) and self.device.type != "cpu"  # FP16 supported on limited backends with CUDA
        if self.pt or self.jit:
            self.model.model.half() if self.half else self.model.model.float()
        bs = 1  # batch_size
        cudnn.benchmark = True  # set True to speed up constant image size inference
        self.model.warmup()  # warmup        
        
        # Initialize subscribers for left and right camera topics
        self.image_sub_left = rospy.Subscriber(
            "/left_cam/image_raw", Image, self.callback_left, queue_size=1
        )
        self.image_sub_right = rospy.Subscriber(
            "/right_cam/image_raw", Image, self.callback_right, queue_size=1
        )

        # Initialize prediction publishers
        self.pred_pub_left = rospy.Publisher(
            "/yolov5/detections_left", BoundingBoxes, queue_size=10
        )
        self.pred_pub_right = rospy.Publisher(
            "/yolov5/detections_right", BoundingBoxes, queue_size=10
        )
        
        # Initialize image publishers for both left and right
        self.publish_image = rospy.get_param("~publish_image")
        if self.publish_image:
            self.image_pub_left = rospy.Publisher(
                "/yolov5/image_out_left", Image, queue_size=10
            )
            self.image_pub_right = rospy.Publisher(
                "/yolov5/image_out_right", Image, queue_size=10
            )

        # Initialize CV_Bridge
        self.bridge = CvBridge()

    def callback_left(self, data):
        """Process left image for yellow cone detection"""
        self.detect_cones(data, "yellow", self.pred_pub_left, self.image_pub_left)

    def callback_right(self, data):
        """Process right image for blue cone detection"""
        self.detect_cones(data, "blue", self.pred_pub_right, self.image_pub_right)

    def detect_cones(self, data, cone_color, pred_pub, image_pub):
        """Common detection function for left and right cameras"""
        im = self.bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")
        im, im0 = self.preprocess(im)

        # Run inference
        im = torch.from_numpy(im).to(self.device)
        im = im.half() if self.half else im.float()
        im /= 255
        if len(im.shape) == 3:
            im = im[None]

        pred = self.model(im, augment=False, visualize=False)
        pred = non_max_suppression(
            pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det
        )

        # Process predictions 
        det = pred[0].cpu().numpy()

        bounding_boxes = BoundingBoxes()
        bounding_boxes.header = data.header
        bounding_boxes.image_header = data.header
        
        annotator = Annotator(im0, line_width=self.line_thickness, example=str(self.names))
        if len(det):
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

            for *xyxy, conf, cls in reversed(det):
                bounding_box = BoundingBox()
                c = int(cls)
                
                if (cone_color == "yellow" and self.names[c] == "coneY") or \
                   (cone_color == "blue" and self.names[c] == "coneB"):
                    
                    # Fill in bounding box message
                    bounding_box.Class = self.names[c]
                    bounding_box.probability = conf 
                    bounding_box.xmin = int(xyxy[0])
                    bounding_box.ymin = int(xyxy[1])
                    bounding_box.xmax = int(xyxy[2])
                    bounding_box.ymax = int(xyxy[3])

                    bounding_boxes.bounding_boxes.append(bounding_box)

                    # Annotate the image
                    if self.publish_image or self.view_image:  
                        label = f"{self.names[c]} {conf:.2f}"
                        annotator.box_label(xyxy, label, color=colors(c, True))       

            im0 = annotator.result()

        pred_pub.publish(bounding_boxes)

        # Publish & visualize images
        if self.view_image:
            cv2.imshow(cone_color, im0)
            cv2.waitKey(1)  # 1 millisecond
        if self.publish_image:
            image_pub.publish(self.bridge.cv2_to_imgmsg(im0, "bgr8"))

    def preprocess(self, img):
        """
        Preprocess image before running inference
        """
        img0 = img.copy()
        img = np.array([letterbox(img, self.img_size, stride=self.stride, auto=self.pt)[0]])
        img = img[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW
        img = np.ascontiguousarray(img)

        return img, img0 


if __name__ == "__main__":

    check_requirements(exclude=("tensorboard", "thop"))
    
    rospy.init_node("yolov5_detector", anonymous=True)
    detector = Yolov5Detector()
    
    rospy.spin()
