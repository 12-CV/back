from fastapi import FastAPI, WebSocket, File, UploadFile
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from io import BytesIO
import torch
import onnxruntime as ort
import time
import base64
from utils.da_transform import load_image
import argparse

def inference_yw(image, session) -> list:
    height, width = image.shape[:2]
    if width != 640 and height == 640:
        raise Exception("이미지 크기를 640x640으로 맞춰주세요.")
    
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))  # Change data layout from HWC to CHW
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]
    outputs = session.run(output_names, {input_name: image})

    class_ids = outputs[0][0]
    bbox = outputs[1][0]
    scores = outputs[2][0]
    additional_info = outputs[3][0]
    score_threshold = [0.03, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.01]

    metadata = []

    for i, score in enumerate(scores):
        if additional_info[i] >= 0:
            if score > score_threshold[additional_info[i]]:
                metadata.append(bbox[i].tolist())

    return {"bboxes": metadata}

def inference_da(image, session):
    image, (orig_h, orig_w) = load_image(image)
    depth = session.run(None, {"image": image})[0]
    depth = cv2.resize(depth[0, 0], (orig_w, orig_h))

    return depth


print("시작@@@@@")
dummy_image = np.random.randint(0, 1, (640, 640, 3), dtype=np.uint8)
session_yw = ort.InferenceSession("./models/yolo.onnx", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
session_da = ort.InferenceSession("./models/depth_anything_vits14.onnx", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
inference_yw(dummy_image, session_yw)
inference_da(dummy_image, session_da)
print("끝@@@@@")

start_time = time.time()
for i in range(10):
    inference_yw(dummy_image, session_yw)
end_time = time.time()
print(f'{end_time - start_time} yoloworld')


start_time = time.time()
for i in range(10):
    inference_da(dummy_image, session_da)
end_time = time.time()
print(f'{end_time - start_time} depthanything')