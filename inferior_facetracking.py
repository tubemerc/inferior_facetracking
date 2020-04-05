# -*- coding: utf-8 -*-
import cv2
import dlib
import numpy as np

set_parameter = {
                 "input_video"       : 0,
                 "img_background"    : "./img/background.png",
                 "img_baseface"      : "./img/baseface.png",
                 "img_eyebrows"      : "./img/eyebrows.png",
                 "img_eyes"          : "./img/eyes.png",
                 "img_mouth"         : "./img/mouth.png",
                 "img_nose"          : "./img/nose.png",
                 "path_predictor"    : "./tools/helen-dataset.dat",
                 "path_cascade_face" : "./tools/haarcascade_frontalface_alt.xml"
                 #"height"            : 
                 #"width"             : 
                }

class inferior_facetracking:
    def __init__(self, input_video,
                       img_background,
                       img_baseface,
                       img_eyebrows,
                       img_eyes,
                       img_mouth,
                       img_nose,
                       path_predictor,
                       path_cascade_face,
                       height=None, width=None):
        print("Setting......")
        self._background  = cv2.imread(img_background)
        self.baseface    = cv2.imread(img_baseface)
        self.eyebrows    = cv2.imread(img_eyebrows)
        self.eyes        = cv2.imread(img_eyes)
        self.mouth       = cv2.imread(img_mouth)
        self.nose        = cv2.imread(img_nose)
        self.height      = height
        self.width       = width
        # predictor, detector, cascade classifier設定
        self.predictor  = dlib.shape_predictor(path_predictor)
        if self.predictor:
            print("predictor done.")
        self.detector = dlib.get_frontal_face_detector()
        if self.detector:
            print("detector done.")
        self.face_cascade = cv2.CascadeClassifier(path_cascade_face)
        if self.face_cascade:
            print("face_cascade done.")
        # 映像入力
        self.org = cv2.VideoCapture(input_video)
        self.end_flag, self.frame_org = self.org.read()
        # 画像サイズ設定
        _height, _width, _ = self.frame_org.shape
        print("<Original>\n "
              "Height :", _height, "\n",
              "Width  :", _width)
        if height is None:
            self.height = _height
        if width is None:
            self.width = _width
        print("<Setting>\n "
              "Height       :", self.height,       "\n",
              "Width        :", self.width,        "\n",
              "Predictor    :", path_predictor,    "\n",
              "Face_cascade :", path_cascade_face)
        # 背景を画面サイズに合わせる
        self._background = cv2.resize(self._background, dsize=(self.width, self.height))
        self.background = self._background.copy()
        
        cv2.namedWindow("Landmarks")
        cv2.namedWindow("IFTRK")
        print("Complete.")
    
    def face_tracking(self):
        flag_tracking = False
        tracker = cv2.TrackerMedianFlow_create()
        
        while self.end_flag:
            self.frame_gray = cv2.cvtColor(self.frame_org, cv2.COLOR_BGR2GRAY)
            # 顔領域更新
            if flag_tracking == True:
                print("tracker update.")
                flag_tracking, face_area = tracker.update(self.frame_org)
                #print(flag_tracking)
                self.face_area = [list(map(int, face_area))]
                #print(self.face_area)
                self.face_landmarks()
                # 顔パーツ配置
                if self.landmarks:
                    self.put_eyes()
                    self.put_mouth()
                    self.put_eyebrows()
                    self.put_nose()
                # 顔領域表示
                for x, y, w, h in self.face_area:
                    cv2.rectangle(self.frame_org,
                                  (x, y),
                                  (x + w, y + h),
                                  (0, 0, 255),
                                  thickness = 1)
            # 顔領域抽出
            elif flag_tracking == False:
                detected_face = self.face_cascade.detectMultiScale(self.frame_gray,
                                                                   scaleFactor  = 1.1,
                                                                   minNeighbors = 3,
                                                                   minSize      = (50, 50))
                #print(len(detected_face))
                if len(detected_face) == 1:
                    for x, y, w, h in detected_face:
                        face_area = (x, y, w, h)
                        #print(face_area)
                        print("tracker initializing.")
                        # 2回目のinitが上手くいかない（なぜ？）
                        flag_tracking = tracker.init(self.frame_org, face_area)
                        #print(flag_tracking)
                    
                    self.face_area = [list(map(int, face_area))]
                    self.face_landmarks()
                    
                    for x, y, w, h in self.face_area:
                        cv2.rectangle(self.frame_org,
                                      (x, y),
                                      (x + w, y + h),
                                      (0, 0, 255),
                                      thickness=2)
                elif len(detected_face) == 0:
                    print("[ERROR] couldn not detect face.")
                else:
                    print("[ERROR] detect multiple face.")
            
            cv2.imshow("Landmarks", self.frame_org)
            cv2.imshow("IFTRK", self.background)
            
            self.background = self._background.copy()
            self.end_flag, self.frame_org = self.org.read()
            
            # ESCキーで終了.
            key = cv2.waitKey(1)  # 1ms
            if key == 27:
                break
        
        cv2.destroyAllWindows()
        self.org.release()
    
    # landmarks判別
    def face_landmarks(self):
        for face in self.face_area:
            (x, y, w, h) = face
            face_rects = self.detector(self.frame_gray, 1)
            self.landmarks = []
            for rect in face_rects:
                self.landmarks.append(np.array(
                        [[p.x, p.y] for p in self.predictor(self.frame_gray, rect).parts()]))
        #print(self.landmarks)
        # landmarks表示
        for landmark in self.landmarks:
                for points in landmark:
                    cv2.drawMarker(self.frame_org,
                                   (points[0], points[1]),
                                   (0, 255, 0))
        
        i = 133
        cv2.drawMarker(self.frame_org, (self.landmarks[0][i][0], self.landmarks[0][i][1]), (255, 0, 0))
    
    # 目配置
    def put_eyes(self):
        _eye_l_w = self.landmarks[0][29][0] - self.landmarks[0][18][0]
        _eye_l_h = self.landmarks[0][34][1] - self.landmarks[0][25][1]
        if _eye_l_w > 0 and _eye_l_h > 0:
            re_eye_l = cv2.resize(self.eyes, dsize=(_eye_l_w, _eye_l_h))
            self.background[self.landmarks[0][25][1]:self.landmarks[0][25][1] + _eye_l_h,
                            self.landmarks[0][18][0]:self.landmarks[0][18][0] + _eye_l_w]\
                = re_eye_l
        else:
            print("[ERROR] negative position of left eye.")
        
        _eye_r_w = self.landmarks[0][40][0] - self.landmarks[0][51][0]
        _eye_r_h = self.landmarks[0][56][1] - self.landmarks[0][45][1]
        if _eye_r_w > 0 and _eye_r_h > 0:
            re_eye_r = cv2.resize(self.eyes, dsize=(_eye_r_w, _eye_r_h))
            self.background[self.landmarks[0][45][1]:self.landmarks[0][45][1] + _eye_r_h,
                            self.landmarks[0][51][0]:self.landmarks[0][51][0] + _eye_r_w]\
                = re_eye_r
        else:
            print("[ERROR] negative position of right eye.")
    
    # 口配置
    def put_mouth(self):
        _mouth_w = self.landmarks[0][164][0] - self.landmarks[0][148][0]
        _mouth_h = self.landmarks[0][10][1] - self.landmarks[0][187][1]
        if _mouth_w > 0 and _mouth_h > 0:
            re_mouth = cv2.resize(self.mouth, dsize=(_mouth_w, _mouth_h))
            self.background[self.landmarks[0][187][1]:self.landmarks[0][187][1] + _mouth_h,
                            self.landmarks[0][148][0]:self.landmarks[0][148][0] + _mouth_w]\
                = re_mouth
        else:
            print("[ERROR] negative position of mouth.")
        
    # 眉配置
    def put_eyebrows(self):
        _brow_l_w = self.landmarks[0][73][0] - self.landmarks[0][62][0]
        _brow_l_h = self.landmarks[0][78][1] - self.landmarks[0][67][1]
        if _brow_l_w > 0 and _brow_l_h > 0:
            re_brow_l = cv2.resize(self.eyebrows, dsize=(_brow_l_w, _brow_l_h))
            self.background[self.landmarks[0][67][1]:self.landmarks[0][67][1] + _brow_l_h,
                            self.landmarks[0][62][0]:self.landmarks[0][62][0] + _brow_l_w]\
                = re_brow_l
        else:
            print("[ERROR] negative position of left eyebrow.")
        
        _brow_r_w = self.landmarks[0][84][0] - self.landmarks[0][95][0]
        _brow_r_h = self.landmarks[0][100][1] - self.landmarks[0][89][1]
        if _brow_r_w > 0 and _brow_r_h > 0:
            re_brow_r = cv2.resize(self.eyebrows, dsize=(_brow_r_w, _brow_r_h))
            self.background[self.landmarks[0][89][1]:self.landmarks[0][89][1] + _brow_r_h,
                            self.landmarks[0][95][0]:self.landmarks[0][95][0] + _brow_r_w]\
                = re_brow_r
        else:
            print("[ERROR] negative position of right eyebrow.")
            
    # 鼻配置
    def put_nose(self):
        _nose_w = self.landmarks[0][144][0] - self.landmarks[0][133][0]
        _nose_h = self.landmarks[0][138][1] - self.landmarks[0][130][1]
        if _nose_w > 0 and _nose_h > 0:
            re_nose = cv2.resize(self.nose, dsize=(_nose_w, _nose_h))
            self.background[self.landmarks[0][133][1]:self.landmarks[0][133][1] + _nose_h,
                            self.landmarks[0][148][0]:self.landmarks[0][148][0] + _nose_w]\
                = re_nose
        else:
            print("[ERROR] negative position of nose.")

if __name__ == "__main__":
    live2d = inferior_facetracking(**set_parameter)
    live2d.face_tracking()