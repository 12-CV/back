import cv2
import numpy as np
from fastapi import WebSocket

from logic.Render import *

def convert_canvas_to_image(canvas):
    width, height = canvas.get_width_height()
    canvas_array = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape((height, width, 3))
    canvas_image = cv2.cvtColor(canvas_array, cv2.COLOR_RGB2BGR)

    return canvas_image

async def on_receive_video(websocket: WebSocket):
    await websocket.accept()
    canvas = CustomCanvas()

    while True:
        frame_buffer = await websocket.receive_bytes()
        metadata = await websocket.receive_json()

        frame = cv2.imdecode(np.frombuffer(frame_buffer, np.uint8), cv2.IMREAD_COLOR)
        processed_metadata, beep, frame_count = process_metadata(frame, metadata)

        render(processed_metadata, frame, canvas)

        img_encoded = cv2.imencode('.jpg', frame)[1].tobytes()
        canvas_image = convert_canvas_to_image(canvas)
        canvas_img_encoded = cv2.imencode('.jpg', canvas_image)[1].tobytes()

        await websocket.send_bytes(img_encoded)
        await websocket.send_bytes(canvas_img_encoded)
        await websocket.send_json({
            "beep": beep,
            "frame_count": frame_count
        })
