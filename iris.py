from operator import rshift
import cv2 as cv
import numpy as np
import os
import mediapipe as mp
from skimage.filters import (threshold_otsu, threshold_niblack,
                             threshold_sauvola)

model_sr = r"D:\teco_nico\D3_HackStack-main\D3_HackStack-main\ESPCN_x4.pb"
modelName = model_sr.split(os.path.sep)[-1].split("_")[0].lower()
modelScale = model_sr.split("_x")[-1]
modelScale = int(modelScale[:modelScale.find(".")])
sr = cv.dnn_superres.DnnSuperResImpl_create()
sr.readModel(model_sr)
sr.setModel(modelName, modelScale)

mp_face_mesh = mp.solutions.face_mesh
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
# right eyes indices
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
# LEFT_IRIS = [474, 475, 476, 477]
# RIGHT_IRIS = [469, 470, 471, 472]
LEFT_IRIS = [474]
RIGHT_IRIS = [469]
cap = cv.VideoCapture(0)
with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
) as face_mesh:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv.flip(frame, 1)
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        img_h, img_w = frame.shape[:2]
        results = face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            # print(results.multi_face_landmarks[0].landmark)
            mesh_points = np.array(
                [np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])
            print(mesh_points.shape)
            cv.polylines(frame, [mesh_points[LEFT_IRIS]], True, (0,255,0), 1, cv.LINE_AA)
            cv.polylines(frame, [mesh_points[RIGHT_IRIS]], True, (0,255,0), 1, cv.LINE_AA)

            # draw lines
            # for i in range(16):
            #     cv.line(frame, mesh_points[RIGHT_EYE[i-1]],mesh_points[RIGHT_EYE[i]],(0,255,0), 1)
            # for i in range(16):
            #     cv.line(frame, mesh_points[LEFT_EYE[i-1]],mesh_points[LEFT_EYE[i]],(0,255,0), 1)

            # creating mask
            mask = np.zeros(frame.shape[0:2], dtype=np.uint8)
            l_points = np.array([np.array(mesh_points[LEFT_EYE])])
            r_points = np.array([np.array(mesh_points[RIGHT_EYE])])
            # method 1 smooth region
            cv.drawContours(mask, [l_points,r_points], -1, (255, 255, 255), -1, cv.LINE_AA)
            # method 2 not so smooth region
            # cv.fillPoly(mask, points, (255))
            res = cv.bitwise_and(frame, frame, mask=mask)
            rect = cv.boundingRect(l_points)  # returns (x,y,w,h) of the rect
            rect = cv.boundingRect(r_points)  # returns (x,y,w,h) of the rect
            cropped = res[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]

            ## crate the white background of the same size of original image
            wbg = np.ones_like(frame, np.uint8) * 255
            cv.bitwise_not(wbg, wbg, mask=mask)
            # overlap the resulted cropped image on the white background
            dst = wbg + res
            cropped = sr.upsample(cropped)
            cropped = cv.cvtColor(cropped, cv.COLOR_RGB2GRAY)
            cropped = cv.equalizeHist(cropped)

            image = cropped

            # image = cv.cvtColor(image,cv.COLOR_RGB2GRAY)
            print("Shape of cropped window",image.shape)
            # Smoothing
            kernel = np.ones((3, 3), np.float32) / 9
            image = cv.morphologyEx(image, cv.MORPH_OPEN, kernel)
            # image = cv.filter2D(image, -1, kernel)
            # image = cv.blur(image, (5, 5))
            # image = cv.GaussianBlur(image, (5, 5), 0)
            # image = cv.medianBlur(image, 3)
            image = cv.bilateralFilter(image,3, 75, 75)

            # THRESHOLDING
            # sauvola = threshold_sauvola(image, window_size=25)
            # adaptive = cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 15, -2)
            _, binary = cv.threshold(image, 100, 255, cv.THRESH_BINARY)
            _, otsu = cv.threshold(image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
            contours, hierarchy = cv.findContours(otsu, 1, 2)
            otsu = cv.resize(otsu,(100,50),interpolation=cv.INTER_NEAREST)
            print(len(contours))
            for i in contours:
                area = cv.contourArea(i)
                if area > 1000:
                    rect = cv.minAreaRect(i)
                    box = cv.boxPoints(rect)
                    box = np.int0(box)
                    # otsu = cv.drawContours(otsu, [box], -1, (0, 255, 0), 2)
                    print(area)

                else:
                    epsilon = 0.1 * cv.arcLength(i, True)
                    approx = cv.approxPolyDP(i, epsilon, True)
                    otsu =cv.fillPoly(otsu, [approx], color=(255, 255, 255))
            # Black pixel count
            count = cv.countNonZero(otsu)
            print(" zero count", otsu.size - count)


        cv.imshow('Frame', frame)
        # cv.imshow("mask",mask)
        # cv.imshow('res',res)
        # cv.imshow('new', cropped)
        cv.imshow('Threshold',otsu)

        key = cv.waitKey(1)
        if key == ord('q'):
            break
cap.release()
cv.destroyAllWindows()
