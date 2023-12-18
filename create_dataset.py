import os
import mediapipe as mp
import cv2
import pickle

import matplotlib.pyplot as plt

mediapipe_hands = mp.solutions.hands
mediapipe_drawing = mp.solutions.drawing_utils
mediapipe_drawing_styles = mp.solutions.drawing_styles

hands = mediapipe_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)


DATA_DIR = './sign-data'

data = []
labels = []

for dir_ in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        if img is not None:  # Check if the image was successfully loaded
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            results = hands.process(img_rgb)
            if results.multi_hand_landmarks:
                for landmarks in results.multi_hand_landmarks:
                   for i in range(len(landmarks.landmark)):
                       x = landmarks.landmark[i].x
                       y = landmarks.landmark[i].y
                       data_aux.append(x)
                       data_aux.append(y)
                labels.append(dir_)
                data.append(data_aux)
        else:
            print(f"Failed to load image: {img_path}")

f = open('data.pickle', 'wb')
pickle.dump({"data" : data, "labels" : labels}, f)
f.close()

