import cv2 as cv
import mediapipe as mp
import time


class FaceMeshDetector():
    def __init__(
                self, staticMode = False,
                maxFaces = 1,
                minDetectionCon = 0.5,
                minTrackCon = 0.5
                ):
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon

        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(
                                                # self.staticMode,
                                                # self.maxFaces,
                                                # self.minDetectionCon,
                                                # self.minTrackCon
                                                )
        self.mpDraw = mp.solutions.drawing_utils
        self.drawSpec = self.mpDraw.DrawingSpec(thickness = 1, circle_radius = 1, color=(255, 0, 0))

    def findFaceMesh(self, img, draw= True):
        imgRgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

        self.result = self.faceMesh.process(imgRgb)

        faces = []
        if self.result.multi_face_landmarks:
            
            for faceLms in self.result.multi_face_landmarks:
                # self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_CONTOURS, self.drawSpec, self.drawSpec)

                face = []
                # ih, iw, ic = img.shape
                for id, lm in enumerate(faceLms.landmark):
                    # print(lm)
                    ih, iw, ic = img.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    # print(id, x, y)
                    face.append([x, y])
                faces.append(face)
        return faces, img














class FaceDetector():
    def __init__(self, minDetectioncon = 0.75):
        self.minDetectioncon = minDetectioncon

        self.mpFaceDetection = mp.solutions.face_detection
        self.faceDetection = self.mpFaceDetection.FaceDetection()
        self.mpDraw = mp.solutions.drawing_utils

    # when we give argument to the python function we are actually giving the refrence of that variable not the copy of it
    def findFaces(self, img, draw = True):
        imgRgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.result = self.faceDetection.process(imgRgb)
        bboxs = []

        if self.result.detections:
            for id, detection in enumerate(self.result.detections):
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, ic = img.shape
                bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                        int(bboxC.width * iw), int(bboxC.height * ih)
                bboxs.append([id, bbox, detection.score])
                # cv.rectangle(img, bbox, (255, 0, 255), 2)
                # img = self.fancyDraw(img, bbox)
                # cv.putText(img, f'{int(detection.score[0] * 100)}', (bbox[0], bbox[1] - 20), cv.FONT_HERSHEY_PLAIN, fontScale=2, color=(255, 0, 255), thickness=2)
        return img, bboxs


    def fancyDraw(self, img, bbox, l = 30, thickness = 5, rthickness = 1):
        x, y, w, h = bbox
        x1, y1 = x+w, y+w
        cv.rectangle(img, bbox, (255, 0, 255), rthickness)
        # top left
        cv.line(img, (x, y), (x+l, y), color = (255, 0, 255), thickness=thickness)
        cv.line(img, (x, y), (x, y+l), color = (255, 0, 255), thickness=thickness)
        # top right
        cv.line(img, (x1, y), (x1 - l, y), color = (255, 0, 255), thickness=thickness)
        cv.line(img, (x1, y), (x1, y+l), color = (255, 0, 255), thickness=thickness)
        return img
