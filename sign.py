# Sign Language to Text and Voice Translator (Basic Prototype)

import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import kagglehub
# Download latest version
path = kagglehub.dataset_download("datamunge/sign-language-mnist")

print("Path to dataset files:", path)
from tensorflow.keras.models import load_model

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# Load pre-trained sign language model (for static hand signs A-Z)
model = load_model("asl_model.h5")
classes = [chr(i) for i in range(65, 91)]  # A-Z

# MediaPipe hands setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False,
                        max_num_hands=1,
                        min_detection_confidence=0.7,
                        min_tracking_confidence=0.7)

# Initialize webcam
cap = cv2.VideoCapture(0)

sentence = ""
last_pred = ""

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            coords = []
            for lm in hand_landmarks.landmark:
                coords.append(lm.x)
                coords.append(lm.y)

            coords = np.array(coords).reshape(1, -1)
            pred = model.predict(coords)
            pred_class = np.argmax(pred)
            letter = classes[pred_class]

            if letter != last_pred:
                sentence += letter
                last_pred = letter

    # Display output
    cv2.putText(frame, f"Text: {sentence}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow("Sign Language Translator", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        # Clear the sentence
        sentence = ""
    elif key == ord('s'):
        # Speak the sentence
        engine.say(sentence)
        engine.runAndWait()

cap.release()
cv2.destroyAllWindows()