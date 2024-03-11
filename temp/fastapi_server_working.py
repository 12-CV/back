from fastapi import FastAPI, WebSocket, File, UploadFile
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from io import BytesIO
import torch
import onnxruntime as ort
from time import time
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

app = FastAPI()

async def on_receive_video(websocket: WebSocket):
    # WebSocket 연결 수락
    await websocket.accept()

    while True:
        data = await websocket.receive_bytes()
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        metadata = inference(image, session)

        await websocket.send_json(metadata)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="yoloworld", help='Choose between "yoloworld" or "depthanything"')
    args = parser.parse_args()

    print(f"{args.model} 추론 서버를 켜고 있습니다...")

    if args.model == "yoloworld":
        inference = inference_yw
        model_path = './models/yolo.onnx'
    elif args.model == "yoloworld":
        inference = inference_da
        model_path = './models/depth_anything_vits14.onnx'
    else:
        pass

    dummy_image = np.random.randint(0, 1, (640, 640, 3), dtype=np.uint8)
    session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    inference(dummy_image, session)

    app.add_websocket_route("/ws", on_receive_video)
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=30348, log_level="info", ws="auto", lifespan="on")
    