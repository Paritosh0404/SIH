import cv2
import datetime
import tkinter as tk
from tkinter import Label, Frame, GROOVE
from PIL import Image, ImageTk
import pyttsx3
import mediapipe as mp
import numpy as np
import tensorflow as tf

# Initialize the MediaPipe Hand module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Load the gesture classification model
class KeyPointClassifier:
    def __init__(self, model_path='model/keypoint_classifier/keypoint_classifier.tflite', num_threads=1):
        self.interpreter = tf.lite.Interpreter(model_path=model_path, num_threads=num_threads)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def classify(self, landmark_list):
        input_tensor = np.array([landmark_list], dtype=np.float32)
        self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor)
        self.interpreter.invoke()
        output_tensor = self.interpreter.get_tensor(self.output_details[0]['index'])
        gesture_id = np.argmax(np.squeeze(output_tensor))
        return gesture_id

# Initialize the classifier
gesture_classifier = KeyPointClassifier()

# Gesture labels (you can replace these with actual gesture names)
gesture_labels = ["Open Hand", "Closed Fist", "Thumbs Up", "Thumbs Down", "Peace Sign"]

# Initialize the Tkinter window
win = tk.Tk()
width = win.winfo_screenwidth()
height = win.winfo_screenheight()
win.geometry(f"{width}x{height}")
win.title('Gesture Interpreter and Speech Translator (GIST)')

# Create frames and labels
frame_1 = Frame(win, width=width, height=height, bg="#181823").place(x=0, y=0)
title_label = Label(win, text='Gesture Interpreter and Speech Translator (GIST)', font=('Comic Sans MS', 26, 'bold'), bd=5, bg='#20262E',
                    fg='#F5EAEA', relief=GROOVE, width=5000)
title_label.pack(pady=20, padx=200)

# Clock and Date
clock = Label(win, font=("Arial", 20), relief=GROOVE, width=15, bd=5, fg="#F5EAEA", bg="#20262E")
clock.pack(anchor=tk.NW, padx=150, pady=10)
clock.place(x=70, y=320)

cal = Label(win, font=("Arial", 20), relief=GROOVE, width=15, bd=5, fg="#F5EAEA", bg="#20262E")
cal.pack(anchor=tk.NW, padx=150, pady=10)
cal.place(x=70, y=400)

def update_clock():
    now = datetime.datetime.now()
    clock.config(text=now.strftime("%H:%M:%S"))
    clock.after(1000, update_clock)

update_clock()
cal.config(text=datetime.date.today().strftime("%B %d, %Y"))

# Define voice function
def voice():
    engine = pyttsx3.init()
    engine.say(CountGesture.get())
    engine.runAndWait()

# Exit function
def exit_app():
    cv2.destroyAllWindows()
    win.destroy()

# Add exit button
exit_button = tk.Button(win, text='Exit', command=exit_app, font=('Arial', 14), bg='#20262E', fg='#F5EAEA')
exit_button.place(x=width - 150, y=50)

# Add sound button
sound_button = tk.Button(win, text='Sound', command=voice, font=('Arial', 14), bg='#20262E', fg='#F5EAEA')
sound_button.place(x=width - 150, y=100)

# Current gesture display
CountGesture = tk.StringVar()
crr_label = Label(win, text='Current Gesture:', font=('Calibri', 18, 'bold'), bd=5, bg='#20262E', width=15,
                  fg='#F5EAEA', relief=GROOVE)
status_label = Label(win, textvariable=CountGesture, font=('Georgia', 18, 'bold'), bd=5, bg='#20262E', width=30,
                     fg='#F5EAEA', relief=GROOVE)
status_label.place(x=520, y=700)
crr_label.place(x=200, y=700)

# Main camera loop
label1 = Label(win)
label1.place(x=width // 2 - 300, y=100)

# Calculate landmark list from hand landmarks
def calc_landmark_list(image, hand_landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []

    for landmark in hand_landmarks.landmark:
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])

    # Flatten the list for the model input
    flatten_landmark_point = np.array(landmark_point).flatten()
    return flatten_landmark_point

# Main camera loop function
def select_img():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame to mirror the display
        frame = cv2.flip(frame, 1)

        # Process the frame and detect hands
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(frame_rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Calculate the landmark list
                landmark_list = calc_landmark_list(frame, hand_landmarks)

                # Classify the gesture
                gesture_id = gesture_classifier.classify(landmark_list)

                # Get the corresponding gesture label
                gesture_text = gesture_labels[gesture_id] if gesture_id < len(gesture_labels) else "Unknown Gesture"

                # Set the current gesture display in Tkinter
                CountGesture.set(gesture_text)

        # Convert the frame for displaying in Tkinter
        framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(framergb)
        finalImage = ImageTk.PhotoImage(image)
        label1.configure(image=finalImage)
        label1.image = finalImage

        # Update the GUI
        win.update_idletasks()
        win.update()

        # Small delay for smoother video
        win.after(1)

    cap.release()

# Start the camera loop
win.after(1, select_img)

win.mainloop()
