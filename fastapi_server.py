from fastapi import FastAPI, WebSocket, File, UploadFile
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from io import BytesIO
import torch
import onnxruntime as ort
from time import time
import base64


app = FastAPI()

async def on_receive_video(websocket: WebSocket):
    # WebSocket 연결 수락
    await websocket.accept()

    while True:
        data = await websocket.receive_bytes()
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        image = cv2.resize(image, (640, 640))
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0)

        # 모델 추론
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

        # 결과를 클라이언트로 전송
        await websocket.send_json({"bboxes": metadata})

if __name__ == "__main__":
    session = ort.InferenceSession('./yolow-l.onnx', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.add_websocket_route("/ws", on_receive_video)
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=30300, log_level="info", ws="auto", lifespan="on")  