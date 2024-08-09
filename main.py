import cv2 as cv
import numpy as np
from media_pipe_modules import FaceMeshDetector, FaceDetector
import time
import math



# reading a video
capture = cv.VideoCapture(0)

# creating object from the modules
mpFaceMesh = FaceMeshDetector()
faceDetector = FaceDetector()

isTrue = True
angle = 0
theta_i = 0
theta_j = 6
while isTrue:
    success, frame = capture.read()
    fliped_frame = cv.flip(frame, 1)

    # get the face mesh
    faces, img = mpFaceMesh.findFaceMesh(fliped_frame)
    img, bboxs = faceDetector.findFaces(fliped_frame)

    nose_x, nose_y= faces[0][1]
    x_min, y_min, w, h = bboxs[0][1]
    center_x, center_y = x_min + int(w/2), y_min + int(h/2)

    # dealing with angle
    if(nose_x > center_x and nose_y < center_y):
        angle = math.atan(abs(nose_x-center_x)/abs(nose_y-center_y))
        angle = math.degrees(angle) + 0
    elif(nose_x > center_x and nose_y > center_y):
        angle = math.atan(abs(nose_y-center_y)/abs(nose_x-center_x))
        angle = math.degrees(angle) + 90
    elif(nose_x < center_x and nose_y > center_y):
        angle = math.atan(abs(nose_x-center_x)/abs(nose_y-center_y))
        angle = math.degrees(angle) + 180
    elif(nose_x < center_x and nose_y < center_y):
        angle = math.atan(abs(nose_y-center_y)/abs(nose_x-center_x))
        angle = math.degrees(angle) + 270

    x1 = abs(nose_x)
    x2 = abs(center_x)
    y1 = abs(nose_y)
    y2 = abs(center_y)
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    if(angle > theta_i and angle < theta_j and distance > 8):

        print("inside save")
        cv.imwrite(f'personB/frame{theta_j}.jpg', fliped_frame[y_min:y_min+h, x_min:x_min+w])
        theta_i += 6 
        theta_j += 6
    
    if(theta_j > 360):
        break



    # angle = math.degrees(angle) + 90

    cv.circle(img=fliped_frame, center=(nose_x, nose_y), radius=2, color=(0, 0, 255), thickness=1)
    cv.circle(img=fliped_frame, center=(center_x, center_y), radius=2, color=(0, 0, 255), thickness=1)
    cv.putText(img=fliped_frame, text=str(angle), org=(100, 100), fontFace=cv.FONT_HERSHEY_TRIPLEX, color=(255, 255, 255),
           fontScale=2, thickness=2)

    cv.imshow("image", img)
    if cv.waitKey(17) == ord('d'):
        break

capture.release()
cv.destroyAllWindows()