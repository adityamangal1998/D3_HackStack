from tkinter import *
from tkinter import messagebox
from PIL import Image, ImageTk
import ctypes
from tkinter import filedialog
from tkinter.filedialog import asksaveasfile
from tkinter import ttk
import cv2
import mediapipe as mp
import math
import sys
import numpy as np
from typing import List, Mapping, Optional, Tuple, Union
from mediapipe.framework.formats import detection_pb2
from mediapipe.framework.formats import location_data_pb2

# ctypes.windll.shcore.SetProcessDpiAwareness(1)

root = Tk()
root.resizable(False, False)
root.geometry("1120x960")
root.title("D3")
root.iconbitmap('ico.ico')


# title label
title_label = Label(root, bg='#611417', text="D3",
                    justify=LEFT, anchor=W, font='Bahnschrift 35', fg='white')
title_label.place(x=0, y=0, width=1120, height=80)


#  main Label
main_frame = Frame(root, bg='#333333')
main_frame.place(x=0, y=80, height=880, width=1120)

# original image window

bg_orig = Image.open('nvid.png')
bg_orig = bg_orig.resize((640, 510), Image.ANTIALIAS)
bg_orig = ImageTk.PhotoImage(bg_orig)
original_window = Label(main_frame, image=bg_orig)
original_window.place(x=20, y=10, height=510, width=640)

# face cam window

bg_face = Image.open('face.png')
bg_face = bg_face.resize((300, 300), Image.ANTIALIAS)
bg_face = ImageTk.PhotoImage(bg_face)
face_window = Label(main_frame, image=bg_face)
face_window.place(x=680, y=10, height=300, width=300)

# outline cam window

bg_outline = Image.open('face.png')
bg_outline = bg_outline.resize((300, 300), Image.ANTIALIAS)
bg_outline = ImageTk.PhotoImage(bg_outline)
outline_window = Label(main_frame, image=bg_outline)
outline_window.place(x=20, y=540, height=300, width=300)

# mesh cam window

bg_mesh = Image.open('face.png')
bg_mesh = bg_mesh.resize((300,
                          300), Image.ANTIALIAS)
bg_mesh = ImageTk.PhotoImage(bg_mesh)
mesh_window = Label(main_frame, image=bg_mesh)
mesh_window.place(x=360, y=540, height=300, width=300)

# _________________________________________________________


def display():
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_face_mesh = mp.solutions.face_mesh
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils
    radius = 2
    color = (0, 255, 0)
    thickness = -1
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def _normalized_to_pixel_coordinates(
            normalized_x: float, normalized_y: float, image_width: int,
            image_height: int) -> Union[None, Tuple[int, int]]:
        """Converts normalized value pair to pixel coordinates."""

        # Checks if the float value is between 0 and 1.
        def is_valid_normalized_value(value: float) -> bool:
            return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                              math.isclose(1, value))

        if not (is_valid_normalized_value(normalized_x) and
                is_valid_normalized_value(normalized_y)):
            # TODO: Draw coordinates even if it's outside of the image bounds.
            return None
        x_px = min(math.floor(normalized_x * image_width), image_width - 1)
        y_px = min(math.floor(normalized_y * image_height), image_height - 1)
        return x_px, y_px

    def get_roi_face(image: np.ndarray,
                     detection: detection_pb2.Detection):
        image_rows, image_cols, _ = image.shape

        location = detection.location_data
        if location.format != location_data_pb2.LocationData.RELATIVE_BOUNDING_BOX:
            raise ValueError(
                'LocationData must be relative for this drawing funtion to work.')
        # Draws keypoints.
        # for keypoint in location.relative_keypoints:
        #   keypoint_px = _normalized_to_pixel_coordinates(keypoint.x, keypoint.y,
        #                                                  image_cols, image_rows)
        #   cv2.circle(image, keypoint_px, keypoint_drawing_spec.circle_radius,
        #              keypoint_drawing_spec.color, keypoint_drawing_spec.thickness)
        # # Draws bounding box if exists.
        if not location.HasField('relative_bounding_box'):
            return
        relative_bounding_box = location.relative_bounding_box
        rect_start_point = _normalized_to_pixel_coordinates(
            relative_bounding_box.xmin, relative_bounding_box.ymin, image_cols,
            image_rows)
        rect_end_point = _normalized_to_pixel_coordinates(
            relative_bounding_box.xmin + relative_bounding_box.width,
            relative_bounding_box.ymin + relative_bounding_box.height, image_cols,
            image_rows)
        print(f"rect_start_point : {rect_start_point}")
        print(f"rect_end_point : {rect_end_point}")
        return rect_start_point, rect_end_point
        # cv2.rectangle(image, rect_start_point, rect_end_point,
        #               bbox_drawing_spec.color, bbox_drawing_spec.thickness)

    # For webcam input:
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    with mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.5) as face_detection:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # face_image = np.ones((300, 300, 3)) * 255
            face_image = np.zeros([300, 300, 3], dtype=np.uint8)
            face_image.fill(255)
            face_image_1 = np.zeros([300, 300, 3], dtype=np.uint8)
            face_image_1.fill(255)
            results = face_detection.process(image)

            # Draw the face detection annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            img_orginal = image.copy()
            if results.detections:
                for detection in results.detections:
                    rect_start_point, rect_end_point = get_roi_face(
                        image, detection)
                    # try:
                    x1, y1 = rect_start_point[0], rect_start_point[1]
                    x2, y2 = rect_end_point[0], rect_end_point[1]
                    img_orginal = img_orginal[y1 - 70:y2 + 30, x1 - 30:x2 + 30]
                    img_orginal = cv2.resize(img_orginal, (300, 300))
                    face_image = np.zeros([300, 300, 3], dtype=np.uint8)
                    face_image.fill(255)
                    face_image_1 = np.zeros([300, 300, 3], dtype=np.uint8)
                    face_image_1.fill(255)
                    results = face_mesh.process(img_orginal.copy())
                    img_h, img_w, img_c = img_orginal.shape
                    if results.multi_face_landmarks:
                        for face_landmarks in results.multi_face_landmarks:
                            mp_drawing.draw_landmarks(
                                image=face_image,
                                landmark_list=face_landmarks,
                                connections=mp_face_mesh.FACEMESH_CONTOURS,
                                landmark_drawing_spec=None,
                                connection_drawing_spec=mp_drawing_styles
                                .get_default_face_mesh_contours_style())
                            mp_drawing.draw_landmarks(
                                image=face_image_1,
                                landmark_list=face_landmarks,
                                connections=mp_face_mesh.FACEMESH_TESSELATION,
                                landmark_drawing_spec=None,
                                connection_drawing_spec=mp_drawing_styles
                                .get_default_face_mesh_tesselation_style())
                            for idx, lm in enumerate(face_landmarks.landmark):
                                if idx == 1:
                                    # nose
                                    x, y = int(lm.x * img_w), int(lm.y * img_h)
                                    center = (x, y)
                                    left_eye = [x, y]
                                    thickness = 5
                                    img_orginal = cv2.circle(
                                        img_orginal, center, radius, (255, 0, 0), thickness)
                    cv2.circle(img_orginal, (150, 150),
                               radius, color, thickness)

                    # except Exception as e:
                    #     print(f"error : {e}")
                    #     pass
                    # mp_drawing.draw_detection(image, detection)
            # Flip the image horizontally for a selfie-view display.

                original = cv2.flip(image, 1)
                face = cv2.flip(img_orginal, 1)
                outline = face_image
                mesh = cv2.flip(face_image_1, 1)
                # ________________________________________________
                # original image
                original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
                original_array = ImageTk.PhotoImage(Image.fromarray(original))
                original_window['image'] = original_array
                # face image
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face_array = ImageTk.PhotoImage(Image.fromarray(face))
                face_window['image'] = face_array
                # mesh image
                mesh = cv2.cvtColor(mesh, cv2.COLOR_BGR2RGB)
                mesh_array = ImageTk.PhotoImage(Image.fromarray(mesh))
                mesh_window['image'] = mesh_array
                # outline image
                outline = cv2.cvtColor(outline, cv2.COLOR_BGR2RGB)
                outline_array = ImageTk.PhotoImage(Image.fromarray(outline))
                outline_window['image'] = outline_array
                root.update()
                # ________________________________________________
        #     cv2.imshow('MediaPipe Face Detection', face)
        #     cv2.imshow('Original Image', original)
        #     cv2.imshow('Outline', face_image)
        #     cv2.imshow('Mesh', mesh)
        print("____________display end________")

    cap.release()


# __________________________________________________________
# start Button
print("button start")
start_icon = Image.open('str.png')
start_icon = start_icon.resize((130, 60), Image.ANTIALIAS)
start_icon = ImageTk.PhotoImage(start_icon)
start_btn = Button(title_label, image=start_icon, cursor="hand2",
                   activebackground="#611417", command=display, bg='#611417', relief='flat')
start_btn.place(x=950, y=10, width=130, height=60)
print("button stop")
# # exit Button
# exit_icon = Image.open('exit.png')
# exit_icon = exit_icon.resize((130, 60), Image.ANTIALIAS)
# exit_icon = ImageTk.PhotoImage(exit_icon)
# exitt_btn = Button(title_label, image=exit_icon, cursor="hand2",
#                    activebackground="#611417", command=sys.exit, bg='#611417', relief='flat')
# start_btn.place(x=950, y=10, width=130, height=60)
# ________________________________________________________

root.mainloop()
