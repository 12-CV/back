from fastapi import FastAPI, WebSocket
import cv2
import numpy as np
import torch
import onnxruntime as ort
import time
import math
import argparse
from utils.da_transform import load_image
from matplotlib.patches import Circle
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

class MyMplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=6.4, height=6.4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        fig.set_facecolor('black')
        self.axes.tick_params(axis='both', colors='white', labelcolor='white')
        self.clear_axes()

        super(MyMplCanvas, self).__init__(fig)

    def clear_axes(self):
        self.axes.clear()
        self.axes.set_xlim(-10, 10)
        self.axes.set_ylim(0, 20)
        self.axes.set_facecolor('black')

        # 왼쪽과 오른쪽 spine 숨기기
        self.axes.spines['top'].set_visible(False)  # 상단 spine 숨기기
        self.axes.spines['bottom'].set_position('zero')  # 하단 spine을 0 위치로 이동
        self.axes.spines['left'].set_visible(False)  # 왼쪽 spine 숨기기
        self.axes.spines['right'].set_position('zero')  # 오른쪽 spine을 가운데로 이동
        # Y 축 눈금을 오른쪽에 표시
        self.axes.yaxis.tick_right()

        # 중점으로부터 5, 10 거리의 위험 반경 표시
        theta = np.linspace(0, 2*np.pi, 100)
        x = 5 * np.cos(theta)
        y = 5 * np.sin(theta)
        self.axes.plot(x, y, color='white', linestyle='--', marker='', linewidth=0.7)

        x = 10 * np.cos(theta)
        y = 10 * np.sin(theta)
        self.axes.plot(x, y, color='white', linestyle='--', marker='', linewidth=0.7)
        # self.axes.plot([-30, 0, 30], [20, 0, 20], color='white', linestyle='--', marker='', linewidth=0.7)


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

def update_figure(metadata, frame, canvas, blue_face:bool, draw_bbox:bool):
        canvas.clear_axes()
        for bbox in metadata:
            x1, y1, x2, y2 = bbox[:4]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            class_id = bbox[4]
            median_point = bbox[5]
            max_point = bbox[6]                     
            mean_point = bbox[7] 
            middle_point = bbox[8]
            x = bbox[9]
            y = bbox[10]
            rad = bbox[11]
            distance = bbox[12]

            if class_id != 9:
                if distance < 5:
                    stat = 'Danger'
                    color = (0, 0, 255)
                    color_str = "red"

                elif distance < 10:
                    stat = 'Warning'
                    color = (0, 165, 255)
                    color_str = "orange"

                else:
                    stat = 'Safe'
                    color = (0, 255, 0)
                    color_str = "green"

                circle = Circle(xy=(x, y), radius=rad, edgecolor=color_str, facecolor=color_str)
                canvas.axes.add_patch(circle)

                # 거리 20 이내의 객체만 Bbox를 그려준다.
                if distance < 20 and draw_bbox:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, stat, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            else:
                if blue_face:
                    mosaic_area = frame[y1:y2, x1:x2]
                    X, Y = x1//30, y1//30
                    if X <= 0:
                        X = 1
                    if Y <= 0:
                        Y = 1
                    mosaic_area = cv2.resize(mosaic_area, (X,Y))
                    mosaic_area = cv2.resize(mosaic_area, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
                    frame[y1:y2, x1:x2] = mosaic_area
            
        canvas.draw()


async def on_receive_video(websocket: WebSocket):
    await websocket.accept()

    canvas = MyMplCanvas()

    while True:
        image_buffer = await websocket.receive_bytes()
        metadata = await websocket.receive_json()
        frame_count = metadata["frame_count"]
        blur_face = metadata["blur_face"]
        draw_bbox = metadata["draw_bbox"]
        collision_warning = metadata["collision_warning"]

        image = cv2.imdecode(np.frombuffer(image_buffer, np.uint8), cv2.IMREAD_COLOR)
        bboxes = inference_yw(image, session_yw)
        depth = inference_da(image, session_da)
        
        metadata = []
        beep = False
        for bbox in bboxes:
            bbox_depth_region = depth[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
            if bbox_depth_region.size == 0:
                continue

            median_point = np.median(bbox_depth_region)
            max_point = np.max(bbox_depth_region)                        
            mean_point = np.mean(bbox_depth_region)      
            middle_point = depth[int((bbox[1] + bbox[3]) / 2)][int((bbox[0] + bbox[2]) / 2)]
            bbox += [float(median_point), float(max_point), float(mean_point), float(middle_point)]

            # 클라이언트 렌더링을 서버에서 하도록 설정
            x1, y1, x2, y2 = bbox[:4]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            corr = 15

            # 객체 x축의 중간값을 각도로 수정 (-45 ~ 45도)
            r = ((x1 + x2) / 2 - 320) * (45 / 320)
            r = math.radians(r)

            # median 값을 실제 거리로 근사
            median_depth = 21 - (median_point * 4 / 3) + corr
            
            # depth와 각도에 따른 x, y값
            x = median_depth * math.sin(r)
            y = median_depth * math.cos(r) - corr

            # x축 너비의 따른 원 크기 조절
            rad = (x2 - x1) / 160

            distance = (x ** 2 + y ** 2) ** 0.5 - rad
            if distance < 5 and collision_warning:
                beep = True

            if y < 0 and distance < 10:
                y = -np.log(-y + 1) + 0.7

            bbox.extend([x, y, rad, distance])
            
            metadata.append(bbox)

        update_figure(metadata, image, canvas, blur_face, draw_bbox)
        img_encoded = cv2.imencode('.jpg', image)[1].tobytes()

        width, height = canvas.get_width_height()
        canvas_array = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape((height, width, 3))
        canvas_image = cv2.cvtColor(canvas_array, cv2.COLOR_RGB2BGR)
        canvas_img_encoded = cv2.imencode('.jpg', canvas_image)[1].tobytes()

        await websocket.send_bytes(img_encoded)
        await websocket.send_bytes(canvas_img_encoded)
        await websocket.send_json({
            "beep": beep,
            "frame_count": frame_count
        })

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