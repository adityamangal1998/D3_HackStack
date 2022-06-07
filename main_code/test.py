# import time
# import cv2
# import mediapipe as mp
# from utils import debug
# import calculate
# import core
# import numpy as np
# # import glass_main
# import head_main
# import utils
# import head_hack
# import yawn_and_eye
# from Graph import Graph
# params = {
#     'glass_text': 'glass',
#     'head_text_1': 'processing head',
#     'head_text_2': 'processing head',
#     'person_text': 'processing person',
#     'eye_ratio': 'processing eye',
#     'mouth_ratio': 'processing mouth',
#     'eye_close_flag': False,
#     'eye_close_time_stamp': 0,
#     'eye_time_stamp': [],
#     'mouth_time_stamp': [],
#     'mouth_open_time_stamp': 0,
#     'head_down_time_stamp': 0,
#     'mouth_open_flag': False,
#     'head_down_flag': False,
#     'eye_blink_stamp': [],
#     'mouth_yawn_stamp': [],
#     'head_time_stamp': [],
#     'ref_epoch': 0,
#     'result_eye': "",
#     'result_mouth': "",
#     'result_head': ""
# }
# mp_face_detection = mp.solutions.face_detection
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles
# mp_face_mesh = mp.solutions.face_mesh
#
# face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
#
# scale_graph = 5
# scale_graph_line = 50
# graph_eye = Graph(60*scale_graph, 60*scale_graph)
# graph_mouth = Graph(60*scale_graph, 60*scale_graph)
#
#
# with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
#     cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
#     while cap.isOpened():
#         success, image = cap.read()
#         original_frame = image.copy()
#         cropped_face = np.zeros([300, 300, 3], dtype=np.uint8)
#         cropped_face.fill(255)
#         facemesh_tesselation = np.zeros([300, 300, 3], dtype=np.uint8)
#         facemesh_tesselation.fill(255)
#         facemesh_contours = np.zeros([300, 300, 3], dtype=np.uint8)
#         facemesh_contours.fill(255)
#         head_img = np.zeros([300, 300, 3], dtype=np.uint8)
#         head_img.fill(255)
#         canvas = np.zeros([250, 800, 3], dtype=np.uint8)
#         canvas.fill(0)
#         graph_eye_canvas = np.zeros([60*scale_graph,60*scale_graph, 3], dtype=np.uint8)
#         graph_eye_canvas.fill(0)
#         graph_mouth_canvas = np.zeros([60 * scale_graph, 60 * scale_graph, 3], dtype=np.uint8)
#         graph_mouth_canvas.fill(0)
#         eye_frame = np.zeros([300, 150, 3], dtype=np.uint8)
#         eye_frame.fill(255)
#         mouth_frame = np.zeros([300, 150, 3], dtype=np.uint8)
#         mouth_frame.fill(255)
#         try:
#             results = face_detection.process(image)
#             img_original = image.copy()
#             if results.detections:
#                 for detection in results.detections:
#                     rect_start_point, rect_end_point = utils.get_roi_face(image, detection)
#                     x1, y1 = rect_start_point[0], rect_start_point[1]
#                     x2, y2 = rect_end_point[0], rect_end_point[1]
#                     img_original = img_original[y1 - 100:y2 + 100, x1 - 90:x2 + 90]
#                     img_original = cv2.resize(img_original, (300, 300))
#                     params['mouth_ratio'], params['eye_ratio'],eye_frame,mouth_frame = yawn_and_eye.main(img_original.copy(),eye_frame,mouth_frame)
#                     eye_frame = cv2.flip(cv2.cvtColor(eye_frame, cv2.COLOR_BGR2RGB), 1)
#                     mouth_frame = cv2.flip(cv2.cvtColor(mouth_frame, cv2.COLOR_BGR2RGB), 1)
#
#
#                     graph_eye.update_frame(int(params['eye_ratio']/2)*scale_graph_line)
#                     graph_eye_canvas = graph_eye.get_graph()
#                     graph_eye_canvas[:150,:] = eye_frame
#                     graph_eye_canvas = cv2.line(graph_eye_canvas, (0, 200), (300, 200), (0, 0, 255), thickness=2)
#
#                     graph_mouth.update_frame(int(params['mouth_ratio'] / 11) * scale_graph_line)
#                     graph_mouth_canvas = graph_mouth.get_graph()
#                     graph_mouth_canvas[:150, :] = mouth_frame
#                     graph_mouth_canvas = cv2.line(graph_mouth_canvas, (0,200), (300,200), (0,0,255), thickness=2)
#
#                     params['head_text_1'], head_img, facemesh_tesselation, facemesh_contours = head_hack.main(
#                         img_original.copy())
#                     params['head_text_2'], head_x, head_y = head_main.main(original_frame.copy())
#                     if params['head_text_1'] == "processing head" and params['head_text_2'] == "Looking Up":
#                         params['head_text_1'] = "Looking Up"
#                     # graph_mouth
#                     print(f"eye ratio : {int(params['eye_ratio']/1.5)*scale_graph_line}")
#                     # params = core.eye_main(params)
#                     # params = calculate.eye_calculate_main(params)
#                     # params = core.mouth_main(params)
#                     # params = calculate.mouth_calculate_main(params)
#                     # params = core.head_main(params)
#                     # params = calculate.head_calculate_main(params)
#                     original_frame = cv2.flip(original_frame, 1)
#         except Exception as e:
#             print(f"error : {e}")
#             params['head_text'] = 'processing head'
#             params['glass_text'] = 'processing glass'
#             params['person_text'] = 'processing person'
#             params['eye_ratio'] = 'processing eye'
#             params['mouth_ratio'] = 'processing mouth'
#
#         cv2.putText(canvas, "head -> " + params['head_text_1'], (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#         cv2.putText(canvas, "wearing-> " + params['glass_text'], (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
#         cv2.putText(canvas, "eye ratio -> " + str(params['eye_ratio']), (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 1,
#                     (0, 0, 255), 2)
#         cv2.putText(canvas, "mouth ratio -> " + str(params['mouth_ratio']), (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 1,
#                     (0, 0, 255), 2)
#         cv2.imshow('canvas', canvas)
#         cv2.imshow('head image', head_img)
#         cv2.imshow('person', original_frame)
#         cv2.imshow('facemesh_tesselation', cv2.flip(facemesh_tesselation, 1))
#         cv2.imshow('facemesh_contours', cv2.flip(facemesh_contours, 1))
#         cv2.imshow('eye_graph_canvas', graph_eye_canvas)
#         cv2.imshow('mouth_graph_canvas', graph_mouth_canvas)
#         if cv2.waitKey(5) & 0xFF == 27:
#             break
#     cap.release()
from datetime import datetime

now = datetime.now()

current_time = datetime.today().strftime("%I:%M:%S %p")
print("Current Time =", current_time)