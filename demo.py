import cv2
from deepface import DeepFace
import pygame
import os
import time
import numpy as np

# Initialize Pygame
pygame.init()

# Create the window
screen_width = 800
screen_height = 400
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Emotion-based Music Player")

# Set colors
white = (255, 255, 255)
black = (0, 0, 0)
gray = (200, 200, 200)

# Initialize font module
pygame.font.init()
font = pygame.font.Font(None, 36)
path = str(os.getcwd())

# Emotion to MP3 file mapping
emotion_to_mp3 = {
    'happy': path + '/happy.mp3',
    'sad': path + '/sad.mp3',
}

def play_music(emotion):
    if emotion in emotion_to_mp3:
        mp3_file = emotion_to_mp3[emotion]
        if os.path.exists(mp3_file):
            pygame.mixer.music.load(mp3_file)
            pygame.mixer.music.play()
        else:
            print(f"MP3 file not found for emotion: {emotion}")

def draw_emotion_text(emotion):
    text_surface = font.render(emotion, True, white)
    text_rect = text_surface.get_rect(center=(screen_width // 2, screen_height // 2))
    screen.blit(text_surface, text_rect)

def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    return cv2.LUT(image, table)

# Load face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start capturing video
cap = cv2.VideoCapture(0)

# Initialize emotion variable with a default value
emotion = 'happy'
current_music = None
music_playing = False
window_size = 30
expression_list = []
sad_detected = False
happy_detected = False

running = True
while running:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Check if frame is captured successfully
    if not ret:
        print("Failed to capture frame from the webcam.")
        break

    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Convert grayscale frame to RGB format
    rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    facesize = 0
    for (x,y,w,h) in faces:
        facesize = max(facesize, x*y)

    dummy_face = []
    for (x,y,w,h) in faces:
        if x*y == facesize:
            dummy_face.append((x,y,w,h))

    for (x, y, w, h) in dummy_face:
        # Extract the face ROI (Region of Interest)

        gamma = 1.0  # change the value here to get different result
        rgb_frame = adjust_gamma(rgb_frame, gamma=gamma)
        frame = adjust_gamma(frame, gamma=gamma)
        face_roi = rgb_frame[y:y + h, x:x + w]

        try:
            # Perform emotion analysis on the face ROI
            result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)

            # Determine the dominant emotion
            dominant_emotion = result[0]['dominant_emotion']
            expression_list.append(dominant_emotion)
            emotion = max(set(expression_list), key=expression_list.count)

            if len(expression_list) > window_size:
                expression_list.pop(0)

            # Map the emotion to either happy or sad
            #if emotion in ['neutral', 'sad', 'unhappy']:
            if emotion != 'happy':
                emotion = 'sad'
                sad_detected = True
                happy_detected = False
            else:
                emotion = 'happy'
                sad_detected = False
                happy_detected = True

            # Play the corresponding MP3 song based on the detected emotion
            if current_music != emotion and not music_playing:
                play_music(emotion)
                current_music = emotion
                music_playing = True
            elif sad_detected and current_music == 'happy':
                pygame.mixer.music.fadeout(2000)  # Fade out the happy music over 2 seconds
                time.sleep(2)  # Wait for the fade out to complete
                play_music('sad')
                current_music = 'sad'
                music_playing = True
            elif happy_detected and current_music == 'sad':
                pygame.mixer.music.fadeout(2000)  # Fade out the sad music over 2 seconds
                time.sleep(2)  # Wait for the fade out to complete
                play_music('happy')
                current_music = 'happy'
                music_playing = True

            # Draw rectangle around face and label with predicted emotion
            emotion_display = 'happy' if emotion=='happy' else 'not happy'
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, emotion_display, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            cv2.putText(frame, dominant_emotion, (x + 200, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        except Exception as e:
            print(f"Error analyzing emotion: {str(e)}")

    # Display the resulting frame
    cv2.imshow('Real-time Emotion Detection', frame)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Check if the music has finished playing
    if music_playing and not pygame.mixer.music.get_busy():
        music_playing = False

    screen.fill(black)
    draw_emotion_text(emotion)
    pygame.display.flip()

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()
pygame.quit()