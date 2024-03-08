from fastapi import FastAPI, WebSocket, File, UploadFile
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from io import BytesIO
import torch
import onnxruntime as ort
from time import time
import base64

def inference(image, session):
    height, width = image.shape[:2]
    if width != 640 and height == 640:
        raise Exception("이미지 크기를 640x640으로 맞춰주세요.")
    
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))  # Change data layout from HWC to CHW
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]
    return session.run(output_names, {input_name: image})

app = FastAPI()

async def on_receive_video(websocket: WebSocket):
    # WebSocket 연결 수락
    await websocket.accept()

    while True:
        data = await websocket.receive_bytes()
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        outputs = inference(image, session)

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

        # 결과를 클라이언트로 전송
        await websocket.send_json({"bboxes": metadata})

if __name__ == "__main__":
    print("서버를 켜고 있습니다...")
    session = ort.InferenceSession('./models/yolo.onnx', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    dummy_image = np.random.randint(0, 256, (640, 640, 3), dtype=np.uint8)
    inference(dummy_image, session)
    app.add_websocket_route("/ws", on_receive_video)
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=30348, log_level="info", ws="auto", lifespan="on")
    