import torch
import onnxruntime as ort

import asyncio
import websockets
import struct
import time
import cv2
import numpy as np
import queue
import threading
import pickle

recieve_frame_queue = queue.Queue()

def run_session(image, session):
    height, width = image.shape[:2]
    if width != 640 and height == 640:
        raise Exception("이미지 크기를 640x640으로 맞춰주세요.")
    
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))  # Change data layout from HWC to CHW
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]
    return session.run(output_names, {input_name: image})


async def recieve_frame(websocket):
        try:
            while True:
                frame_number_packed = await websocket.recv()
                frame_buffer = await websocket.recv()

                # number는 Integer, metadata는 Json
                frame_number = struct.unpack("I", frame_number_packed)[0]
                frame_np = np.frombuffer(frame_buffer, np.uint8)
                frame = cv2.imdecode(frame_np, cv2.IMREAD_COLOR)

                recieve_frame_queue.put((frame_number, frame))

        except websockets.exceptions.ConnectionClosedError as e:
            print(f"Connection closed: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")

class VideoRecieveThread(threading.Thread):
    def __init__(self):
        super().__init__()

    def run(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        start_server = websockets.serve(recieve_frame, "0.0.0.0", 30348)
        asyncio.get_event_loop().run_until_complete(start_server)
        asyncio.get_event_loop().run_forever()

async def send_frame(websocket, session):
        while True:
            try:
                frame_number, frame = recieve_frame_queue.get(block=True, timeout=1)
            except queue.Empty:
                continue
                
            metadata = run_session(frame, session)
            await websocket.send(struct.pack('I', frame_number))
            metadata_packed = pickle.dumps(metadata)
            await websocket.send(metadata_packed)

class VideoSendThread(threading.Thread):
    def __init__(self, session):
        super().__init__()
        self.session = session

    def run(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        start_server = websockets.serve(lambda websocket: send_frame(websocket, self.session), "0.0.0.0", 30349)
        asyncio.get_event_loop().run_until_complete(start_server)
        asyncio.get_event_loop().run_forever()

if __name__ == '__main__':
    print("서버 준비중...")
    dummy_image = np.random.randint(0, 256, (640, 640, 3), dtype=np.uint8)
    session = ort.InferenceSession('./models/car_yolo.onnx', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    run_session(dummy_image, session)

    video_recieve_thread = VideoRecieveThread()
    video_recieve_thread.start()

    video_send_thread = VideoSendThread(session)
    video_send_thread.start()

    print("서버가 잘 시작되었습니다!")
    # video_recieve_thread.join()
    # video_send_thread.join()