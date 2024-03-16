from fastapi import FastAPI, WebSocket, File, UploadFile
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from io import BytesIO
import torch
import onnxruntime as ort
from time import time
import argparse
from utils.da_transform import load_image


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
                metadata.append(bbox[i].tolist() + [int(additional_info[i])])
    
    return metadata

def inference_da(image, session):
    image, (orig_h, orig_w) = load_image(image)
    depth = session.run(None, {"image": image})[0]
    depth = cv2.resize(depth[0, 0], (orig_w, orig_h))

    return depth

async def on_receive_video(websocket: WebSocket):
    # WebSocket 연결 수락
    await websocket.accept()

    while True:
        data = await websocket.receive_bytes()
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

        bboxes = inference_yw(image, session_yw)
        depth = inference_da(image, session_da)

        metadata = []
        for bbox in bboxes:
            bbox_depth_region = depth[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
            if bbox_depth_region.size == 0:
                continue

            median_point = np.median(bbox_depth_region)
            max_point = np.max(bbox_depth_region)                        
            mean_point = np.mean(bbox_depth_region)      
            middle_point = depth[int((bbox[1] + bbox[3]) / 2)][int((bbox[0] + bbox[2]) / 2)]
            bbox += [float(median_point), float(max_point), float(mean_point), float(middle_point)]
            metadata.append(bbox)
        
        await websocket.send_json({"bboxes": metadata})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=30348, help='enter your available port number!')
    args = parser.parse_args()

    dummy_image = np.random.randint(0, 256, (640, 640, 3), dtype=np.uint8)
    session_yw = ort.InferenceSession('./models/yolow-l.onnx', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    session_da = ort.InferenceSession("./models/depth_anything_vits14.onnx", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    inference_yw(dummy_image, session_yw)
    inference_da(dummy_image, session_da)
    app = FastAPI()
    app.add_websocket_route("/ws", on_receive_video)
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info", ws="auto", lifespan="on")  