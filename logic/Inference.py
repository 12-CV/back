import cv2
import torch
import numpy as np
import onnxruntime as ort

from config import *
from logic.Transform import load_image

session_yw = None
session_da = None

def inference_yw(image) -> list:
    height, width = image.shape[:2]
    if width != 640 and height == 640:
        raise Exception("이미지 크기를 640x640으로 맞춰주세요.")
    
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))  # Change data layout from HWC to CHW
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    input_name = session_yw.get_inputs()[0].name
    output_names = [o.name for o in session_yw.get_outputs()]
    outputs = session_yw.run(output_names, {input_name: image})

    class_ids = outputs[0][0]
    bbox = outputs[1][0]
    scores = outputs[2][0]
    additional_info = outputs[3][0]
    score_threshold = CLASS_THRESHHOLD

    metadata = []

    for i, score in enumerate(scores):
        if additional_info[i] >= 0:
            if score > score_threshold[additional_info[i]]:
                metadata.append(bbox[i].tolist() + [int(additional_info[i])])
    
    return metadata

def inference_da(image):
    image, (orig_h, orig_w) = load_image(image)
    depth = session_da.run(None, {"image": image})[0]
    depth = cv2.resize(depth[0, 0], (orig_w, orig_h))
    
    return depth

def init_onnx_sessions():
    dummy_image = np.random.randint(0, 256, (640, 640, 3), dtype=np.uint8)
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']

    global session_yw, session_da
    session_yw = ort.InferenceSession(YOLO_ONNX_PATH, providers=providers)
    session_da = ort.InferenceSession(DA_ONNX_PATH, providers=providers)
    inference_yw(dummy_image)
    inference_da(dummy_image)