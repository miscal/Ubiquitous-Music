import cv2
from deepface import DeepFace
import os
import numpy as np

def adjust_gamma(image, gamma=1.0):

   invGamma = 1.0 / gamma
   table = np.array([((i / 255.0) ** invGamma) * 255
      for i in np.arange(0, 256)]).astype("uint8")

   return cv2.LUT(image, table)

# Load face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize emotion variable with a default value
#emotion = 'neutral'
current_music = None
music_playing = False
music_start_time = 0
window_size = 1
expression_list = []
expression_list_all = []

frame_path = "frame/frame/sad"
correct_emotion = "sad" # or not happy
gamma = 1.0 # change the value here to get different result


for frame in os.listdir(frame_path):
    frame = os.path.join(frame_path, frame)

    frame = cv2.imread(frame)
    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Convert grayscale frame to RGB format
    rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Extract the face ROI (Region of Interest)
        
        rgb_frame = adjust_gamma(rgb_frame, gamma=gamma)
        face_roi = rgb_frame[y:y + h, x:x + w]


        try:
            # Perform emotion analysis on the face ROI
            result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)

            # Determine the dominant emotion
            dominant_emotion = result[0]['dominant_emotion']
            #if dominant_emotion != "happy":
            #    dominant_emotion = "not happy"
            expression_list.append(dominant_emotion)
            emotion = max(set(expression_list), key=expression_list.count)
            print(dominant_emotion, emotion)

            if len(expression_list) > window_size:
                expression_list.pop(0)

            expression_list_all.append(emotion)

        except Exception as e:
            print(f"Error analyzing emotion: {str(e)}")


cnt = 0
for i in range(len(expression_list_all)):
    if expression_list_all[i] == correct_emotion:
        cnt += 1

print("total accuracy: {}".format(cnt/len(expression_list_all)))