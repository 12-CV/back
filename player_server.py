import asyncio
import websockets
import struct
import time
import cv2
import numpy as np
import queue
import threading
import pickle

from yolov5_detector import YOLOv5Detector

recieve_frame_queue = queue.Queue()
yolo_detector = None

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
        start_server = websockets.serve(recieve_frame, "0.0.0.0", 30121)
        asyncio.get_event_loop().run_until_complete(start_server)
        asyncio.get_event_loop().run_forever()

async def send_frame(websocket):
        while True:
            try:
                frame_number, frame = recieve_frame_queue.get(block=True, timeout=1)
            except queue.Empty:
                continue
                
            # 인퍼런스를 여기서 진행하고
            frame = cv2.resize(frame, (640, 480))
            metadata = yolo_detector.detect(frame=frame, frame_number=frame_number)
            
            await websocket.send(struct.pack('I', frame_number))
            metadata_packed = pickle.dumps(metadata)
            await websocket.send(metadata_packed)

class VideoSendThread(threading.Thread):
    def __init__(self):
        super().__init__()

    def run(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        start_server = websockets.serve(send_frame, "0.0.0.0", 30122)
        asyncio.get_event_loop().run_until_complete(start_server)
        asyncio.get_event_loop().run_forever()


if __name__ == '__main__':
    
    yolo_detector = YOLOv5Detector()
    
    video_recieve_thread = VideoRecieveThread()
    video_recieve_thread.start()

    video_send_thread = VideoSendThread()
    video_send_thread.start()
    # video_recieve_thread.join()
    # video_send_thread.join()