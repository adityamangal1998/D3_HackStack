import cv2
from Graph import Graph
import yawn_and_eye
import head_hack
import utils
import head_main
import numpy as np
import core
import calculate
from utils import debug
import mediapipe as mp
from tkinter import *
from tkinter import messagebox
from PIL import Image, ImageTk
import ctypes
from tkinter import filedialog
from tkinter.filedialog import asksaveasfile
from tkinter import ttk
# import hack
from datetime import datetime
import sys

ctypes.windll.shcore.SetProcessDpiAwareness(1)

root = Tk()
root.resizable(False, False)
root.geometry("1900x980")
root.title("D3")
root.iconbitmap('images/ico.ico')


# title label
bg_title = Image.open('images/title.png')
bg_title = bg_title.resize((1900, 80), Image.ANTIALIAS)
bg_title = ImageTk.PhotoImage(bg_title)
title_label = Label(root,  image=bg_title, bg='black')
title_label.place(x=0, y=0, width=1900, height=80)


#  main Label
bg_main = Image.open('images/bg_p.png')
bg_main = bg_main.resize((1900, 880), Image.ANTIALIAS)
bg_main = ImageTk.PhotoImage(bg_main)
main_frame = Label(root, bg='black')
main_frame.place(x=0, y=80, height=900, width=1900)

# cropped Face image window

bg_orig = Image.open('images/face.png')
bg_orig = bg_orig.resize((300, 300), Image.ANTIALIAS)
bg_orig = ImageTk.PhotoImage(bg_orig)
original_window = Label(main_frame, image=bg_orig)
original_window.place(x=0, y=0, height=300, width=300)

# contour image window

bg_contour = Image.open('images/face.png')
bg_contour = bg_contour.resize((300, 300), Image.ANTIALIAS)
bg_contour = ImageTk.PhotoImage(bg_contour)
contour_window = Label(main_frame, image=bg_contour)
contour_window.place(x=0, y=300, height=300, width=300)

# mesh cam window

bg_mesh = Image.open('images/face.png')
bg_mesh = bg_mesh.resize((300, 300), Image.ANTIALIAS)
bg_mesh = ImageTk.PhotoImage(bg_mesh)
mesh_window = Label(main_frame, image=bg_mesh)
mesh_window.place(x=0, y=600, height=300, width=300)

# canvas cam window

bg_canvas = Image.open('images/canvas.png')
bg_canvas = bg_canvas.resize((800, 300), Image.ANTIALIAS)
bg_canvas = ImageTk.PhotoImage(bg_canvas)
canvas_window = Label(main_frame, image=bg_canvas)
canvas_window.place(x=300, y=600, height=300, width=800)

# headBound cam window

bg_headBound = Image.open('images/face.png')
bg_headBound = bg_headBound.resize((300, 300), Image.ANTIALIAS)
bg_headBound = ImageTk.PhotoImage(bg_headBound)
headBound_window = Label(main_frame, image=bg_headBound)
headBound_window.place(x=1100, y=600, height=300, width=300)

# headposDetail cam window

bg_headPosDetail = Image.open('images/face.png')
bg_headPosDetail = bg_headPosDetail.resize((500, 500), Image.ANTIALIAS)
bg_headPosDetail = ImageTk.PhotoImage(bg_headPosDetail)
headPosDetail_window = Label(main_frame, image=bg_headPosDetail)
headPosDetail_window.place(x=1400, y=600, height=300, width=500)

# person cam window

bg_person = Image.open('images/person.png')
bg_person = bg_person.resize((800, 600), Image.ANTIALIAS)
bg_person = ImageTk.PhotoImage(bg_person)
person_window = Label(main_frame, image=bg_person)
person_window.place(x=300, y=0, height=600, width=800)


# eyeGraph cam window

bg_eyeGraph = Image.open('images/face.png')
bg_eyeGraph = bg_eyeGraph.resize((300, 300), Image.ANTIALIAS)
bg_eyeGraph = ImageTk.PhotoImage(bg_eyeGraph)
eyeGraph_window = Label(main_frame, image=bg_eyeGraph)
eyeGraph_window.place(x=1100, y=0, height=300, width=300)

# mouthGraph cam window

bg_mouthGraph = Image.open('images/face.png')
bg_mouthGraph = bg_mouthGraph.resize((300, 300), Image.ANTIALIAS)
bg_mouthGraph = ImageTk.PhotoImage(bg_mouthGraph)
mouthGraph_window = Label(main_frame, image=bg_mouthGraph)
mouthGraph_window.place(x=1100, y=300, height=300, width=300)


# mouthGraph Detail cam window

bg_mouthGraphDetail = Image.open('images/face.png')
bg_mouthGraphDetail = bg_mouthGraphDetail.resize((500, 500), Image.ANTIALIAS)
bg_mouthGraphDetail = ImageTk.PhotoImage(bg_mouthGraphDetail)
mouthGraphDetail_window = Label(main_frame, image=bg_mouthGraphDetail)
mouthGraphDetail_window.place(x=1400, y=300, height=300, width=500)

# eyeGraph Detail window

bg_eyeGraphDetail = Image.open('images/face.png')
bg_eyeGraphDetail = bg_eyeGraphDetail.resize((500, 500), Image.ANTIALIAS)
bg_eyeGraphDetail = ImageTk.PhotoImage(bg_eyeGraphDetail)
eyeGraphDetail_window = Label(main_frame, image=bg_eyeGraphDetail)
eyeGraphDetail_window.place(x=1400, y=0, height=300, width=500)


# # outline cam window

# bg_outline = Image.open('face.png')
# bg_outline = bg_outline.resize((300, 300), Image.ANTIALIAS)
# bg_outline = ImageTk.PhotoImage(bg_outline)
# outline_window = Label(main_frame, image=bg_outline)
# outline_window.place(x=20, y=540, height=300, width=300)

# # mesh cam window

# bg_mesh = Image.open('face.png')
# bg_mesh = bg_mesh.resize((300,
#                           300), Image.ANTIALIAS)
# bg_mesh = ImageTk.PhotoImage(bg_mesh)
# mesh_window = Label(main_frame, image=bg_mesh)
# mesh_window.place(x=360, y=540, height=300, width=300)

# _________________________________________________________


def display():

    import time

    # import glass_main
    params = {
        'glass_text': 'glass',
        'head_text_1': 'processing head',
        'head_text_2': 'processing head',
        'person_text': 'processing person',
        'eye_ratio': 'processing eye',
        'mouth_ratio': 'processing mouth',
        'eye_close_flag': False,
        'eye_close_time_stamp': 0,
        'eye_time_stamp': [],
        'mouth_time_stamp': [],
        'mouth_open_time_stamp': 0,
        'head_down_time_stamp': 0,
        'mouth_open_flag': False,
        'head_down_flag': False,
        'eye_blink_stamp': [],
        'mouth_yawn_stamp': [],
        'head_time_stamp': [],
        'ref_epoch': 0,
        'result_eye': "",
        'result_mouth': "",
        'result_head': "",
        "last_blink_trigger_time" : "You are Energetic",
        "last_critical_blink_duration" : "You are Energetic",
        "last_yawn_trigger_time": "You are Energetic",
        "last_critical_yawn_duration": "You are Energetic",
        "last_head_down_trigger_time": "You are Energetic",
        "last_critical_head_down_duration": "You are Energetic",
        "list_head_time_stamp" : []
    }
    debug_bool = True
    radius = 2
    color = (0, 255, 0)
    thickness = -1
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_face_mesh = mp.solutions.face_mesh
    start_time_glass = int(time.time())
    face_mesh = mp_face_mesh.FaceMesh(
        min_detection_confidence=0.5, min_tracking_confidence=0.5)
    counter_sleep = 0
    counter_drowsiness = 0
    scale_graph = 5
    scale_graph_line = 50
    graph_eye = Graph(60*scale_graph, 60*scale_graph)
    graph_mouth = Graph(60*scale_graph, 60*scale_graph)
    ft_color = (57, 181, 74)
    with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap.set(3, 800)
        cap.set(4, 600)
        while cap.isOpened():
            success, image = cap.read()
            original_frame = image.copy()
            cropped_face = np.zeros([300, 300, 3], dtype=np.uint8)
            cropped_face.fill(0)
            facemesh_tesselation = np.zeros([300, 300, 3], dtype=np.uint8)
            facemesh_tesselation.fill(0)
            facemesh_contours = np.zeros([300, 300, 3], dtype=np.uint8)
            facemesh_contours.fill(0)
            head_img = np.zeros([300, 300, 3], dtype=np.uint8)
            head_img.fill(0)
            canvas = np.zeros([300, 800, 3], dtype=np.uint8)
            canvas.fill(0)
            head_bound = np.zeros([300, 300, 3], dtype=np.uint8)
            head_bound.fill(0)
            graph_eye_canvas = np.zeros(
                [60 * scale_graph, 60 * scale_graph, 3], dtype=np.uint8)
            graph_eye_canvas.fill(0)
            graph_mouth_canvas = np.zeros(
                [60 * scale_graph, 60 * scale_graph, 3], dtype=np.uint8)
            graph_mouth_canvas.fill(0)
            eye_frame = np.zeros([300, 150, 3], dtype=np.uint8)
            eye_frame.fill(0)
            mouth_frame = np.zeros([300, 150, 3], dtype=np.uint8)
            mouth_frame.fill(0)
            Eye_val_img = np.zeros([300, 600, 3], dtype=np.uint8)
            Eye_val_img.fill(0)
            Mouth_val_img = np.zeros([300, 600, 3], dtype=np.uint8)
            Mouth_val_img.fill(0)
            Head_val_img = np.zeros([300, 600, 3], dtype=np.uint8)
            Head_val_img.fill(0)
            try:
                results = face_detection.process(image)
                img_original = image.copy()
                if results.detections:
                    for detection in results.detections:
                        rect_start_point, rect_end_point = utils.get_roi_face(
                            image, detection)
                        x1, y1 = rect_start_point[0], rect_start_point[1]
                        x2, y2 = rect_end_point[0], rect_end_point[1]
                        img_original = img_original[y1 -
                                                    100:y2 + 100, x1 - 90:x2 + 90]
                        img_original = cv2.resize(img_original, (300, 300))
                        if time.time() - start_time_glass > 10:
                            start_time_glass = time.time()
                            # glass_text = glass_main.main(img_original.copy())
                            # params['glass_text'] = glass_text
                        params['head_text_1'], head_img, facemesh_tesselation, facemesh_contours, head_bound = head_hack.main(
                            img_original.copy())
                        params['head_text_2'], head_x, head_y = head_main.main(
                            original_frame.copy())
                        if params['head_text_1'] == "processing head" and params['head_text_2'] == "Looking Up":
                            params['head_text_1'] = "Looking Up"
                        params['mouth_ratio'], params['eye_ratio'], eye_frame, mouth_frame = yawn_and_eye.main(
                            img_original.copy(), eye_frame, mouth_frame)
                        eye_frame = cv2.flip(cv2.cvtColor(
                            eye_frame, cv2.COLOR_BGR2RGB), 1)
                        mouth_frame = cv2.flip(cv2.cvtColor(
                            mouth_frame, cv2.COLOR_BGR2RGB), 1)

                        graph_eye.update_frame(
                            int(params['eye_ratio'] / 2) * scale_graph_line)
                        graph_eye_canvas = graph_eye.get_graph()
                        graph_eye_canvas[:150, :] = eye_frame
                        graph_eye_canvas = cv2.line(
                            graph_eye_canvas, (0, 200), (300, 200), (0, 0, 255), thickness=2)

                        graph_mouth.update_frame(
                            int(params['mouth_ratio'] / 11) * scale_graph_line)
                        graph_mouth_canvas = graph_mouth.get_graph()
                        graph_mouth_canvas[:150, :] = mouth_frame
                        graph_mouth_canvas = cv2.line(
                            graph_mouth_canvas, (0, 200), (300, 200), (0, 0, 255), thickness=2)

                        params = core.eye_main(params)
                        params = calculate.eye_calculate_main(params)
                        params = core.mouth_main(params)
                        params = calculate.mouth_calculate_main(params)
                        params = core.head_main(params)
                        params = calculate.head_calculate_main(params)
                        original_frame = cv2.flip(original_frame, 1)
                        if params['result_eye'] == 'sleeping' or params['result_head'] == 'sleeping':
                            cv2.putText(original_frame, 'sleeping', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                        (0, 0, 255), 2)
                            counter_sleep = counter_sleep + 1
                        elif params['result_eye'] == "drowsiness" or params['result_mouth'] == "drowsiness":
                            cv2.putText(original_frame, "drowsiness", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                        (0, 0, 255), 2)
                            counter_drowsiness = counter_drowsiness + 1
                        if 100 > counter_sleep > 0:
                            cv2.putText(original_frame, 'sleeping', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                        (0, 0, 255), 2)
                            counter_sleep = counter_sleep + 1
                        elif 100 > counter_drowsiness > 0:
                            cv2.putText(original_frame, "drowsiness", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                        (0, 0, 255), 2)
                            counter_drowsiness = counter_drowsiness + 1
                        if counter_drowsiness > 100:
                            counter_drowsiness = 0
                        if counter_sleep > 100:
                            counter_sleep = 0
            except Exception as e:
                print(f"error : {e}")
                params['head_text'] = 'processing head'
                params['glass_text'] = 'processing glass'
                params['person_text'] = 'processing person'
                params['eye_ratio'] = 'processing eye'
                params['mouth_ratio'] = 'processing mouth'

            # design parameters
            safe_color = (56, 242, 130)
            danger_color = (0, 0, 255)
            font_family = cv2.FONT_HERSHEY_TRIPLEX
            font_size = 0.5

            cv2.putText(canvas, "Head : " + params['head_text_1'],
                        (20, 50), font_family, 1, safe_color, 1)
            cv2.putText(canvas, "wearing : " + params['glass_text'],
                        (20, 100), font_family, 1, safe_color, 1)
            cv2.putText(canvas, "Eye Ratio : " + str(params['eye_ratio']), (20, 150), font_family, 1,
                        safe_color, 1)
            cv2.putText(canvas, "Mouth Ratio : " + str(params['mouth_ratio']), (20, 200), font_family, 1,
                        safe_color, 1)

            # $$____$$   Cal_img
            # Eye

            cv2.putText(Eye_val_img, "Last Blink Trigger Time  : " + params["last_blink_trigger_time"], (50, 30), font_family, font_size, safe_color, 1)
            cv2.putText(Eye_val_img, "Last Critical Blink Duration : " + params["last_critical_blink_duration"], (50, 90), font_family,
                        font_size, safe_color, 1)
            cv2.putText(Eye_val_img, " Blink count : " + str(len(params['eye_blink_stamp'])), (50, 150),
                        font_family,
                        font_size, safe_color, 1)
            # Mouth
            cv2.putText(Mouth_val_img, "Last Yawn Trigger Time  : " + params['last_yawn_trigger_time'], (50, 30),
                        font_family, font_size, safe_color, 1)
            cv2.putText(Mouth_val_img, "Last Critical Yawn Time : " + params["last_critical_yawn_duration"], (50, 90),
                        font_family,
                        font_size, safe_color, 1)
            cv2.putText(Mouth_val_img, " Yawn count : " + str(len(params['mouth_yawn_stamp'])), (50, 150),
                        font_family,
                        font_size, safe_color, 1)
            # Head

            cv2.putText(Head_val_img, "Last Head Down Trigger Time  : " + params["last_head_down_trigger_time"], (50, 30),
                        font_family, font_size, safe_color, 1)
            cv2.putText(Head_val_img, "Last Critical Head Down Time : " + params["last_critical_head_down_duration"], (50, 90),
                        font_family,
                        font_size, safe_color, 1)
            cv2.putText(Head_val_img, "Head Down count : " + str(len(params["list_head_time_stamp"])), (50, 150),
                        font_family,
                        font_size, safe_color, 1)

            # cropped Face image
            cropped_face = head_img
            cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
            cropped_array = ImageTk.PhotoImage(Image.fromarray(cropped_face))
            original_window['image'] = cropped_array

            # contour image
            contour = facemesh_contours
            contour = cv2.flip(contour, 1)
            contour = cv2.cvtColor(contour, cv2.COLOR_BGR2RGB)
            contour_array = ImageTk.PhotoImage(Image.fromarray(contour))
            contour_window['image'] = contour_array

            # mesh image
            mesh = facemesh_tesselation
            mesh = cv2.flip(mesh, 1)
            mesh = cv2.cvtColor(mesh, cv2.COLOR_BGR2RGB)
            mesh_array = ImageTk.PhotoImage(Image.fromarray(mesh))
            mesh_window['image'] = mesh_array

            # canvas image
            canvas = canvas
            canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
            canvas_array = ImageTk.PhotoImage(Image.fromarray(canvas))
            canvas_window['image'] = canvas_array

            # headBound image
            headBound = cv2.flip(head_bound, 1)
            headBound = cv2.cvtColor(headBound, cv2.COLOR_BGR2RGB)
            headBound_array = ImageTk.PhotoImage(Image.fromarray(headBound))
            headBound_window['image'] = headBound_array

            # # headposDetail image
            headPosDetail = cv2.cvtColor(Head_val_img, cv2.COLOR_BGR2RGB)
            headPosDetail_array = ImageTk.PhotoImage(
                Image.fromarray(headPosDetail))
            headPosDetail_window['image'] = headPosDetail_array

            # mouthGraphDetail image
            mouthGraphDetail = cv2.cvtColor(Mouth_val_img, cv2.COLOR_BGR2RGB)
            mouthGraphDetail_array = ImageTk.PhotoImage(
                Image.fromarray(mouthGraphDetail))
            mouthGraphDetail_window['image'] = mouthGraphDetail_array

            # cv2.imshow('mouth',Mouth_val_img)

            # # eyeGraphDetail image
            eyeGraphDetail = cv2.cvtColor(Eye_val_img, cv2.COLOR_BGR2RGB)
            eyeGraphDetail_array = ImageTk.PhotoImage(
                Image.fromarray(eyeGraphDetail))
            eyeGraphDetail_window['image'] = eyeGraphDetail_array

            # person image
            person = original_frame
            person = cv2.cvtColor(person, cv2.COLOR_BGR2RGB)
            person_array = ImageTk.PhotoImage(Image.fromarray(person))
            person_window['image'] = person_array

            # eyeGraph image
            eyeGraph = graph_eye_canvas
            eyeGraph = cv2.cvtColor(eyeGraph, cv2.COLOR_BGR2RGB)
            eyeGraph_array = ImageTk.PhotoImage(Image.fromarray(eyeGraph))
            eyeGraph_window['image'] = eyeGraph_array

            # mouthGraph image
            mouthGraph = graph_eye_canvas
            mouthGraph = cv2.cvtColor(graph_mouth_canvas, cv2.COLOR_BGR2RGB)
            mouthGraph_array = ImageTk.PhotoImage(Image.fromarray(mouthGraph))
            mouthGraph_window['image'] = mouthGraph_array

            root.update()

            ##### UI section Ends ######

            if cv2.waitKey(5) & 0xFF == 27:
                break
        cap.release()


# __________________________________________________________
# start Button


start_icon = Image.open('images/deploy.png')
start_icon = start_icon.resize((130, 60), Image.ANTIALIAS)
start_icon = ImageTk.PhotoImage(start_icon)
start_btn = Button(title_label, command=display, image=start_icon, cursor="hand2",
                   activebackground="black",  bg='black', relief='flat')
start_btn.place(x=1700, y=10, width=130, height=60)
# print("button stop")
# # exit Button
# exit_icon = Image.open('exit.png')
# exit_icon = exit_icon.resize((130, 60), Image.ANTIALIAS)
# exit_icon = ImageTk.PhotoImage(exit_icon)
# exitt_btn = Button(title_label, image=exit_icon, cursor="hand2",
#                    activebackground="#611417", command=sys.exit, bg='#611417', relief='flat')
# start_btn.place(x=950, y=10, width=130, height=60)
# ________________________________________________________

root.mainloop()
