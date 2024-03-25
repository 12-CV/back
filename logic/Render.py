import cv2
import numpy as np
from matplotlib.patches import Circle
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from logic.Inference import inference_da, inference_yw

class CustomCanvas(FigureCanvas):
    def __init__(self, parent=None, width=6.4, height=6.4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        fig.set_facecolor('black')
        self.axes.tick_params(axis='both', colors='white', labelcolor='white')
        self.clear_axes()

        super(CustomCanvas, self).__init__(fig)

    def clear_axes(self):
        self.axes.clear()
        self.axes.set_xlim(-5, 5)
        self.axes.set_ylim(0, 10)
        self.axes.set_facecolor('black')

        # 왼쪽과 오른쪽 spine 숨기기
        self.axes.spines['top'].set_visible(False)  # 상단 spine 숨기기
        self.axes.spines['bottom'].set_position('zero')  # 하단 spine을 0 위치로 이동
        self.axes.spines['left'].set_visible(False)  # 왼쪽 spine 숨기기
        self.axes.spines['right'].set_position('zero')  # 오른쪽 spine을 가운데로 이동
        # Y 축 눈금을 오른쪽에 표시
        self.axes.yaxis.tick_right()
        self.axes.xaxis.set_ticks([])

        # 중점으로부터 5, 10 거리의 위험 반경 표시
        theta = np.linspace(0, 2*np.pi, 100)
        x = 2 * np.cos(theta)
        y = 2 * np.sin(theta)
        self.axes.plot(x, y, color='white', linestyle='--', marker='', linewidth=0.7)

        x = 4 * np.cos(theta)
        y = 4 * np.sin(theta)
        self.axes.plot(x, y, color='white', linestyle='--', marker='', linewidth=0.7)

def render(metadata, frame, canvas:CustomCanvas):
    canvas.clear_axes()
    for bbox in metadata:
        x1, y1, x2, y2 = bbox[:4]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        class_id = bbox[4]
        middle_point = bbox[5]
        x = bbox[6]
        y = bbox[7]
        rad = bbox[8]
        distance = bbox[9]
        blur_face = bbox[10]
        draw_bbox = bbox[11]

        if class_id != 9:
            if distance < 2:
                stat = 'Danger'
                color = (0, 0, 255)
                color_str = "red"

            elif distance < 4:
                stat = 'Warning'
                color = (0, 165, 255)
                color_str = "orange"

            else:
                stat = 'Safe'
                color = (0, 255, 0)
                color_str = "green"

            circle = Circle(xy=(x, y), radius=rad, edgecolor=color_str, facecolor=color_str)
            canvas.axes.add_patch(circle)

            # 거리 8 이내의 객체만 Bbox를 그려준다.
            if distance < 8 and draw_bbox:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, stat, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        else:
            if blur_face:
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

def process_metadata(frame, metadata) -> list:
    depth = inference_da(frame)
    bboxes = inference_yw(frame)

    frame_count = metadata["frame_count"]
    blur_face = metadata["blur_face"]
    draw_bbox = metadata["draw_bbox"]
    collision_warning = metadata["collision_warning"]
    beep = False

    processed_metadata = []

    for bbox in bboxes:
        x1, y1, x2, y2 = bbox[:4]

        bbox_depth_region = depth[int(y1):int(y2), int(x1):int(x2)]
        if bbox_depth_region.size == 0:
            continue

        middle_point = depth[int((y1 + y2) / 2)][int((x1 + x2) / 2)]
        bbox += [float(middle_point)]

        # 실제 거리로 근사
        y = 0.01875 * middle_point ** 2 -0.83062 * middle_point + 10.28521
        
        # y 비례 x값 보정
        x = (x1 + x2 - 640) / 1280 * (y + 3.3) 

        # y 비례 객체 반지름 고정
        rad = (x2 - x1) / 1280 * (y + 2.7)

        distance = (x ** 2 + y ** 2) ** 0.5 - rad
        if distance < 2 and collision_warning:
            beep = True

        bbox.extend([x, y, rad, distance, blur_face, draw_bbox])
        processed_metadata.append(bbox)

    return processed_metadata, beep, frame_count