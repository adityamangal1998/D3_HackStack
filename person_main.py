import os
import json
import cv2
import numpy as np
import pandas as pd
from utils import timeit
DETECTION_MODEL = 'ssd_mobilenet/'
SWAPRB = True

with open(os.path.join('models', DETECTION_MODEL, 'labels.json')) as json_data:
    CLASS_NAMES = json.load(json_data)


class Detector():
    """Class ssd"""

    @timeit
    def __init__(self):
        self.model = cv2.dnn.readNet(
                'models/ssd_mobilenet/ssd_mobilenet_v2_coco_2018_03_29.pbtxt',
                'models/ssd_mobilenet/frozen_inference_graph.pb')
        self.colors = np.random.uniform(0, 255, size=(100, 3))

    @timeit
    def prediction(self, image):
        self.model.setInput(
                cv2.dnn.blobFromImage(image, size=(300, 300), swapRB=SWAPRB))
        output = self.model.forward()
        result = output[0, 0, :, :]
        # print(f"result : {result}")
        return result

    @timeit
    def filter_prediction(self, output, image, conf_th=0.5, conf_class=[]):
        height, width = image.shape[:-1]
        class_name = lambda x: (
            x['class_id'].astype(int).astype(str).replace(CLASS_NAMES)
        )
        print(f"class_name : {class_name}")
        df = pd.DataFrame(
                output,
                columns=[
                    '_', 'class_id', 'confidence', 'x1', 'y1', 'x2', 'y2'])
        df = df.assign(
                x1=lambda x: (x['x1'] * width).astype(int).clip(0),
                y1=lambda x: (x['y1'] * height).astype(int).clip(0),
                x2=lambda x: (x['x2'] * width).astype(int),
                y2=lambda x: (x['y2'] * height).astype(int),
                class_name=lambda x: (
                    x['class_id'].astype(int).astype(str).replace(CLASS_NAMES)
                    ),
                )
        df['label'] = df['class_name']
        df = df[df['confidence'] > conf_th]
        if len(conf_class) > 0:
            df = df[df['class_id'].isin(conf_class)]
        return df

    def draw_boxes(self, image, df):
        txt = "no person"
        for idx, box in df.iterrows():
            x_min, y_min, x_max, y_max = box['x1'], box['y1'], box['x2'], box['y2']
            color = self.colors[int(box['class_id'])]
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
            txt_loc = (max(x_min+2, 0), max(y_min+2, 0))
            txt = box['label']
        return txt

def main(image):
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    detector = Detector()
    output = detector.prediction(image)
    df = detector.filter_prediction(output, image)
    class_name = detector.draw_boxes(image, df)
    if class_name == "person":
        return "person"
    else:
        return "no person"