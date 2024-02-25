import os
import sys
from pathlib import Path
import torch
import numpy as np

ROOT = Path(__file__).resolve().parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.general import (
    check_img_size,
    non_max_suppression,
    scale_boxes,
)
from utils.augmentations import (
    letterbox,
)
from utils.torch_utils import select_device, smart_inference_mode

class YOLOv5Detector:
    def __init__(self, weights=ROOT / "yolov5n.pt", data=ROOT / "data/coco128.yaml",
                 imgsz=(640, 480), conf_thres=0.25, iou_thres=0.45, max_det=1000, device="",
                 classes=None, augment=False, visualize=False, hide_conf=False, half=False,
                 dnn=False, vid_stride=1):
        
        self.weights = weights
        self.data = data
        self.imgsz = imgsz
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.max_det = max_det
        self.agnostic_nms=False,
        self.device = device
        self.classes = classes
        self.augment = augment
        self.visualize = visualize
        self.hide_conf = hide_conf
        self.half = half
        self.dnn = dnn
        self.vid_stride = vid_stride
        self.model = None
        self.warm_up()

    def warm_up(self):
        # Load model
        self.device = select_device(self.device)
        self.model = DetectMultiBackend(self.weights, device=self.device, dnn=self.dnn, data=self.data, fp16=self.half)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt

        bs = 1
        self.imgsz = check_img_size(self.imgsz, s=self.stride)  # check image size
        self.model.warmup(imgsz=(1 if self.pt or self.model.triton else bs, 3, *self.imgsz))  # warmup


    @smart_inference_mode()
    def detect(self, frame, frame_number):
        im = letterbox(frame, self.imgsz)[0]  # padded resize
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)  # contiguous

        results = []

        # dt[0]
        im = torch.from_numpy(im).to(self.model.device)
        im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        pred = self.model(im, augment=self.augment, visualize=self.visualize)
        
        # dt[2]
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms, max_det=self.max_det)
           
        for det in pred:  # per image
            im0 = frame.copy()

            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

            # Write results
            for *xyxy, conf, cls in reversed(det):
                c = int(cls)  # integer class
                label = self.names[c] if self.hide_conf else f"{self.names[c]}"
                confidence = float(conf)
                confidence_str = f"{confidence:.2f}"

                prediction = {
                    "label": label,
                    "bbox": {
                        "pt1": (int(xyxy[0].item()), int(xyxy[1].item())),
                        "pt2": (int(xyxy[2].item()), int(xyxy[3].item()))
                    },
                    "confidence": confidence_str
                }
                results.append(prediction)
        
        results = [{
            "frame_number": frame_number,
            "model": "YOLOv5",
            "predictions": results
        }]
        return results