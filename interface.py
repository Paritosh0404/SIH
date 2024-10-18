import cv2
import datetime
import tkinter as tk
from tkinter import Label, Frame, GROOVE, Text, END
from PIL import Image, ImageTk
import pyttsx3
import mediapipe as mp
import numpy as np
import csv
import copy
from collections import Counter
from collections import deque
import argparse
import itertools
import tensorflow as tf

from utils import CvFpsCalc
from model import KeyPointClassifier
from model import PointHistoryClassifier

class GestureRecognitionApp:
    def __init__(self, window, window_title, model_path='model/keypoint_classifier/keypoint_classifier.tflite', num_threads=1):
        self.window = window
        self.window.title(window_title)
        self.width = self.window.winfo_screenwidth()
        self.height = self.window.winfo_screenheight()
        self.window.geometry(f"{self.width}x{self.height}")
        self.interpreter = tf.lite.Interpreter(model_path=model_path, num_threads=num_threads)
        #self.draw_point_history()
        self.setup_gui()
        self.setup_models()

        self.cap = cv2.VideoCapture(0)
        self.delay = 15
        self.update()

        self.window.mainloop()

    def draw_bounding_rect(use_brect, image, brect):
        if use_brect:
            # Outer rectangle
            cv2.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                         (0, 0, 0), 1)

        return image

    def setup_gui(self):
        frame_1 = Frame(self.window, width=self.width, height=self.height, bg="#181823").place(x=0, y=0)
        title_label = Label(self.window, text='Gesture Interpreter and Speech Translator (GIST)',
                            font=('Comic Sans MS', 26, 'bold'), bd=5, bg='#20262E', fg='#F5EAEA',
                            relief=GROOVE, width=5000)
        title_label.pack(pady=20, padx=200)

        self.clock = Label(self.window, font=("Arial", 20), relief=GROOVE, width=15, bd=5, fg="#F5EAEA", bg="#20262E")
        self.clock.pack(anchor=tk.NW, padx=150, pady=10)
        self.clock.place(x=70, y=320)

        self.cal = Label(self.window, font=("Arial", 20), relief=GROOVE, width=15, bd=5, fg="#F5EAEA", bg="#20262E")
        self.cal.pack(anchor=tk.NW, padx=150, pady=10)
        self.cal.place(x=70, y=400)

        self.update_clock()
        self.cal.config(text=datetime.date.today().strftime("%B %d, %Y"))

        exit_button = tk.Button(self.window, text='Exit', command=self.exit_app, font=('Arial', 14), bg='#20262E',
                                fg='#F5EAEA')
        exit_button.place(x=self.width - 150, y=50)

        sound_button = tk.Button(self.window, text='Speak', command=self.voice, font=('Arial', 14), bg='#20262E',
                                 fg='#F5EAEA')
        sound_button.place(x=self.width - 150, y=100)

        clear_button = tk.Button(self.window, text='Clear', command=self.clear_text, font=('Arial', 14), bg='#20262E',
                                 fg='#F5EAEA')
        clear_button.place(x=self.width - 150, y=150)

        self.CountGesture = tk.StringVar()
        crr_label = Label(self.window, text='Current Gesture:', font=('Calibri', 18, 'bold'), bd=5, bg='#20262E',
                          width=15,
                          fg='#F5EAEA', relief=GROOVE)
        status_label = Label(self.window, textvariable=self.CountGesture, font=('Georgia', 18, 'bold'), bd=5,
                             bg='#20262E', width=30,
                             fg='#F5EAEA', relief=GROOVE)
        status_label.place(x=520, y=700)
        crr_label.place(x=200, y=700)

        self.text_box = Text(self.window, height=2, width=50, font=('Arial', 18), bg='#F5EAEA', fg='#20262E')
        self.text_box.place(x=520, y=750)

        self.canvas = tk.Canvas(self.window, width=600, height=400)
        self.canvas.place(x=self.width // 2 - 300, y=100)

    def setup_models(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )
        self.keypoint_classifier = KeyPointClassifier()
        self.point_history_classifier = PointHistoryClassifier()

        with open('model/keypoint_classifier/keypoint_classifier_label.csv',
                  encoding='utf-8-sig') as f:
            self.keypoint_classifier_labels = csv.reader(f)
            self.keypoint_classifier_labels = [
                row[0] for row in self.keypoint_classifier_labels
            ]
        with open(
                'model/point_history_classifier/point_history_classifier_label.csv',
                encoding='utf-8-sig') as f:
            self.point_history_classifier_labels = csv.reader(f)
            self.point_history_classifier_labels = [
                row[0] for row in self.point_history_classifier_labels
            ]

        self.history_length = 16
        self.point_history = deque(maxlen=self.history_length)
        self.finger_gesture_history = deque(maxlen=self.history_length)

    #def draw_point_history(self, image, point_history):
     #   for index, point in enumerate(point_history):
      #      if point[0] != 0 and point[1] != 0:
       #         cv2.circle(image, (point[0], point[1]), 1 + int(index / 2),
        #                   (152, 251, 152), 2)

         #   return image

    def update_clock(self):
        now = datetime.datetime.now()
        self.clock.config(text=now.strftime("%H:%M:%S"))
        self.clock.after(1000, self.update_clock)

    def voice(self):
        engine = pyttsx3.init()
        engine.say(self.text_box.get("1.0", END))
        engine.runAndWait()

    def exit_app(self):
        self.cap.release()
        self.window.quit()

    def clear_text(self):
        self.text_box.delete("1.0", END)

    def update(self, landmarks=None):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = self.hands.process(image)
            image.flags.writeable = True

            if results.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                      results.multi_handedness):
                    brect = self.calc_
                    # Check if the correct method name is being used
                    brect = self.calc_bounding_rect(image, hand_landmarks)

                    landmark_list = self.calc_landmark_list(image, hand_landmarks)

                    pre_processed_landmark_list = self.pre_process_landmark(
                        landmark_list)
                    pre_processed_point_history_list = self.pre_process_point_history(
                        image, self.point_history)

                    hand_sign_id = self.keypoint_classifier(pre_processed_landmark_list)
                    if hand_sign_id == 2:  # Point gesture
                        self.point_history.append(landmark_list[8])
                    else:
                        self.point_history.append([0, 0])

                    finger_gesture_id = 0
                    point_history_len = len(pre_processed_point_history_list)
                    if point_history_len == (self.history_length * 2):
                        finger_gesture_id = self.point_history_classifier(
                            pre_processed_point_history_list)

                    self.finger_gesture_history.append(finger_gesture_id)
                    most_common_fg_id = Counter(
                        self.finger_gesture_history).most_common()

                    image = self.draw_bounding_rect(True, image, brect)
                    image = self.draw_landmarks(image, landmark_list)
                    image = self.draw_info_text(
                        image,
                        brect,
                        handedness,
                        self.keypoint_classifier_labels[hand_sign_id],
                        self.point_history_classifier_labels[most_common_fg_id[0][0]],
                    )

                    gesture_text = self.keypoint_classifier_labels[hand_sign_id]
                    self.CountGesture.set(gesture_text)

                    current_sentence = self.text_box.get("1.0", END).strip()
                    if current_sentence:
                        self.text_box.insert(END, " " + gesture_text)
                    else:
                        self.text_box.insert(END, gesture_text)

                     #   image = self.draw_point_history(image, self.point_history)

                photo = ImageTk.PhotoImage(image=Image.fromarray(image))
                self.canvas.create_image(0, 0, image=photo, anchor=tk.NW)
                self.canvas.image = photo

            self.window.after(self.delay, self.update)

            def calc_bounding_rect(self, image, hand_landmarks):
                image_width, image_height = image.shape[1], image.shape[0]
                landmark_array = np.empty((0, 2), int)
                for _, landmark in enumerate(landmarks.landmark):
                    landmark_x = min(int(landmark.x * image_width), image_width - 1)
                    landmark_y = min(int(landmark.y * image_height), image_height - 1)
                    landmark_point = [np.array((landmark_x, landmark_y))]
                    landmark_array = np.append(landmark_array, landmark_point, axis=0)
                x, y, w, h = cv2.boundingRect(landmark_array)
                return [x, y, x + w, y + h]

            def calc_landmark_list(self, image, landmarks):
                image_width, image_height = image.shape[1], image.shape[0]
                landmark_point = []
                for _, landmark in enumerate(landmarks.landmark):
                    landmark_x = min(int(landmark.x * image_width), image_width - 1)
                    landmark_y = min(int(landmark.y * image_height), image_height - 1)
                    landmark_point.append([landmark_x, landmark_y])
                return landmark_point

            def pre_process_landmark(self, landmark_list):
                temp_landmark_list = copy.deepcopy(landmark_list)
                base_x, base_y = 0, 0
                for index, landmark_point in enumerate(temp_landmark_list):
                    if index == 0:
                        base_x, base_y = landmark_point[0], landmark_point[1]
                    temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
                    temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y
                temp_landmark_list = list(
                    itertools.chain.from_iterable(temp_landmark_list))
                mean = np.mean(temp_landmark_list)
                std = np.std(temp_landmark_list)
                temp_landmark_list = [(n - mean) / std for n in temp_landmark_list]
                return temp_landmark_list

            def pre_process_point_history(self, image, point_history):
                image_width, image_height = image.shape[1], image.shape[0]
                temp_point_history = copy.deepcopy(point_history)
                base_x, base_y = 0, 0
                for index, point in enumerate(temp_point_history):
                    if index == 0:
                        base_x, base_y = point[0], point[1]
                    temp_point_history[index][0] = (temp_point_history[index][0] - base_x) / image_width
                    temp_point_history[index][1] = (temp_point_history[index][1] - base_y) / image_height
                temp_point_history = list(itertools.chain.from_iterable(temp_point_history))
                mean = np.mean(temp_point_history)
                std = np.std(temp_point_history)
                temp_point_history = [(n - mean) / std for n in temp_point_history]
                return temp_point_history

            def draw_landmarks(self, image, landmark_point):
                if len(landmark_point) > 0:
                    # ... (previous code for thumb, index, middle, and ring fingers remains the same)

                    # Little finger
                    cv2.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                             (0, 0, 0), 6)
                    cv2.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                             (255, 255, 255), 2)
                    cv2.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                             (0, 0, 0), 6)
                    cv2.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                             (255, 255, 255), 2)
                    cv2.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                             (0, 0, 0), 6)
                    cv2.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                             (255, 255, 255), 2)

                    # Palm
                    cv2.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                             (0, 0, 0), 6)
                    cv2.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                             (255, 255, 255), 2)
                    cv2.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                             (0, 0, 0), 6)
                    cv2.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                             (255, 255, 255), 2)
                    cv2.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                             (0, 0, 0), 6)
                    cv2.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                             (255, 255, 255), 2)
                    cv2.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                             (0, 0, 0), 6)
                    cv2.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                             (255, 255, 255), 2)
                    cv2.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                             (0, 0, 0), 6)
                    cv2.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                             (255, 255, 255), 2)
                    cv2.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                             (0, 0, 0), 6)
                    cv2.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                             (255, 255, 255), 2)
                    cv2.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                             (0, 0, 0), 6)
                    cv2.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                             (255, 255, 255), 2)

                    # Key Points
                for index, landmark in enumerate(landmark_point):
                    if index == 0:  # 手首1
                        cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                                   -1)
                        cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
                    if index == 1:  # 手首2
                        cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                                   -1)
                        cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
                    if index == 2:  # 親指：付け根
                        cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                                   -1)
                        cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
                    if index == 3:  # 親指：第1関節
                        cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                                   -1)
                        cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
                    if index == 4:  # 親指：指先
                        cv2.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                                   -1)
                        cv2.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
                    if index == 5:  # 人差指：付け根
                        cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                                   -1)
                        cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
                    if index == 6:  # 人差指：第2関節
                        cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                                   -1)
                        cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
                    if index == 7:  # 人差指：第1関節
                        cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                                   -1)
                        cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
                    if index == 8:  # 人差指：指先
                        cv2.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                                   -1)
                        cv2.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
                    if index == 9:  # 中指：付け根
                        cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                                   -1)
                        cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
                    if index == 10:  # 中指：第2関節
                        cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                                   -1)
                        cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
                    if index == 11:  # 中指：第1関節
                        cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                                   -1)
                        cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
                    if index == 12:  # 中指：指先
                        cv2.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                                   -1)
                        cv2.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
                    if index == 13:  # 薬指：付け根
                        cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                                   -1)
                        cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
                    if index == 14:  # 薬指：第2関節
                        cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                                   -1)
                        cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
                    if index == 15:  # 薬指：第1関節
                        cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                                   -1)
                        cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
                    if index == 16:  # 薬指：指先
                        cv2.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                                   -1)
                        cv2.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
                    if index == 17:  # 小指：付け根
                        cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                                   -1)
                        cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
                    if index == 18:  # 小指：第2関節
                        cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                                   -1)
                        cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
                    if index == 19:  # 小指：第1関節

                            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
                    if index == 20:  # 小指：指先
                        cv2.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                                   -1)
                        cv2.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)

                        return image

                def draw_bounding_rect(self, use_brect, image, brect):
                            if use_brect:
                                # Outer rectangle
                                cv2.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                                              (0, 0, 0), 1)

                            return image

                def draw_info_text(self, image, brect, handedness, hand_sign_text,
                                           finger_gesture_text):
                            cv2.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                                          (0, 0, 0), -1)

                            info_text = handedness.classification[0].label[0:]
                            if hand_sign_text != "":
                                info_text = info_text + ':' + hand_sign_text
                            cv2.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

                            if finger_gesture_text != "":
                                cv2.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
                                cv2.putText(image, "Finger Gesture:" + finger_gesture_text, (10, 60),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
                                            cv2.LINE_AA)

                            return image

                            if __name__ == '__main__':
                                  parser = argparse.ArgumentParser()

                            parser.add_argument("--device", type=int, default=0)
                            parser.add_argument("--width", help='cap width', type=int, default=960)
                            parser.add_argument("--height", help='cap height', type=int, default=540)

                            parser.add_argument('--use_static_image_mode', action='store_true')
                            parser.add_argument("--min_detection_confidence",
                                                help='min_detection_confidence',
                                                type=float,
                                                default=0.7)
                            parser.add_argument("--min_tracking_confidence",
                                                help='min_tracking_confidence',
                                                type=int,
                                                default=0.5)

                            args = parser.parse_args()

                           # app = GestureRecognitionApp(tk.Tk(), "Gesture Recognition App")