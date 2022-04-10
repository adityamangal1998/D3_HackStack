# https://towardsdatascience.com/head-pose-estimation-using-python-d165d3541600
import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)


def main(image):
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = face_mesh.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    img_h, img_w, img_c = image.shape
    face_3d = []
    face_2d = []
    text = "No Person"
    x = ""
    y = ""
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                    if idx == 1:
                        nose_2d = (lm.x * img_w, lm.y * img_h)
                        nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)
                    x, y = int(lm.x * img_w), int(lm.y * img_h)
                    # Get the 2D Coordinates
                    face_2d.append([x, y])
                    # Get the 3D Coordinates
                    face_3d.append([x, y, lm.z])
                    # Convert it to the NumPy array
            face_2d = np.array(face_2d, dtype=np.float64)
            for x,y in face_2d:
                print(f"x : {x} and y : {y}")
                center = (int(x),int(y))
                radius = 2
                color = (0, 255, 0)
                thickness = -1
                cv2.circle(image, center, radius, color, thickness)
            print(f"len of face_2d : {face_2d}")
            print(f"values of face_2d : {face_2d}")
            # Convert it to the NumPy array
            face_3d = np.array(face_3d, dtype=np.float64)
            # The camera matrix
            focal_length = 1 * img_w
            cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                   [0, focal_length, img_w / 2],
                                   [0, 0, 1]])
            # The distortion parameters
            dist_matrix = np.zeros((4, 1), dtype=np.float64)
            # Solve PnP
            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
            # Get rotational matrix
            rmat, jac = cv2.Rodrigues(rot_vec)
            # Get angles
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
            # Get the y rotation degree
            x = angles[0] * 360
            y = angles[1] * 360
            z = angles[2] * 360
            # See where the user's head tilting
            if y < -10:
                text = "Looking Left"
            elif y > 10:
                text = "Looking Right"
            elif x < -1:
                text = "Looking Down"
            elif x > 20:
                text = "Looking Up"
            else:
                text = "Forward"

    # return text,x,y
    return text,x,y
