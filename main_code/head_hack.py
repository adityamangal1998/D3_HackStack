import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5, min_tracking_confidence=0.5)
radius = 2
color = (0, 255, 0)
thickness = 5


def main(img):
    head_pos = "processing head"
    facemesh_tesselation = np.zeros([300, 300, 3], dtype=np.uint8)
    facemesh_tesselation.fill(0)
    facemesh_contours = np.zeros([300, 300, 3], dtype=np.uint8)
    facemesh_contours.fill(0)
    head_bound = np.zeros([300, 300, 3], dtype=np.uint8)
    head_bound.fill(0)
    drawSpecs = mp_drawing.DrawingSpec(
        thickness=0, circle_radius=0, color=(57, 181, 74))
    try:
        results = face_mesh.process(img.copy())
        img_h, img_w, img_c = img.shape
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=facemesh_tesselation,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_tesselation_style())
                mp_drawing.draw_landmarks(
                    image=facemesh_contours,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_contours_style())

            for face_landmarks in results.multi_face_landmarks:
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx == 1:
                        # nose
                        x, y = int(lm.x * img_w), int(lm.y * img_h)
                        center = (x, y)
                        img = cv2.circle(img, center, radius,
                                         (255, 0, 0), thickness)
                img = cv2.flip(img, 1)

                ##### Head Bound Code ########

                ###############################
                if (125 < y < 175 and x < 125) or (y > 175 and x < 125 and y >= x) or (125 > y >= x and x < 125):
                    head_pos = "Looking Right"
                    cv2.putText(img, "Looking Right", (20, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                elif (125 < y < 175 and x > 175) or (175 < y <= x and x > 175) or (y < 125 and x > 175 and y <= x):
                    head_pos = "Looking Left"
                    cv2.putText(img, "Looking Left", (20, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                elif (125 < x < 175 and y > 175) or (175 < x <= y and y > 175) or (x < 125 and y > 175 and y >= x):
                    head_pos = "Looking Down"
                    cv2.putText(img, "Looking Down", (20, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                elif (125 < x < 175 and y < 125) or (x > 175 and y < 125 and y <= x) or (125 > x >= y and y < 125):
                    head_pos = "Looking Up"
                    cv2.putText(img, "Looking Up", (20, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                elif 125 < y < 175 and 125 < x < 175:
                    head_pos = "Forward"
                    cv2.putText(img, "Forward", (20, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.circle(img, (150, 150), radius, color, thickness)
        return head_pos, img, facemesh_tesselation, facemesh_contours
    except Exception as e:
        print(f"error in head : {e}")
        return head_pos, img, facemesh_tesselation, facemesh_contours
