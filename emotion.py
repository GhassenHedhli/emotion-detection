# import dependencies
import os
import subprocess
import tempfile
import time
from datetime import datetime

import cv2
from deepface import DeepFace
from gtts import gTTS

faceCascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def list_to_dict(predictions):
    # Extracting column names
    column_names = list(predictions[0].keys())

    # Creating a dictionary with column names as keys
    data_dict = {
        column: [entry[column] for entry in predictions] for column in column_names
    }

    # return the resulting dictionary
    return data_dict


def text_to_speech(text, temp_filename):
    # Convert text to speech
    tts = gTTS(text=text, lang="en")
    tts.save(temp_filename)

    # Play the saved audio file
    subprocess.run(["start", temp_filename], shell=True)

    # Wait until the audio file is played
    time.sleep(4)


def delete_mp3_files(temp_files):
    for temp_file in temp_files:
        if os.path.exists(temp_file):
            os.remove(temp_file)


cap = cv2.VideoCapture(1)

# Check if the webcam is opened correctly
if not cap.isOpened():
    cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise IOError("Cannot open webcam")

temp_files = []

try:
    while True:
        ret, frame = cap.read()
        detections = []

        result = DeepFace.analyze(
            frame, actions=["emotion", "age", "gender"], enforce_detection=False
        )

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        faces = faceCascade.detectMultiScale(gray, 1.1, 4)

        # Draw rectangle around the face
        for x, y, w, h in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        font = cv2.FONT_HERSHEY_SIMPLEX

        # list_to_dict
        data_dict = list_to_dict(result)

        # Extract emotion, age, and gender
        emotion = data_dict["dominant_emotion"][0]
        age = data_dict["age"]
        gender = data_dict["dominant_gender"][0]

        label = f"Emotion: {emotion}, Age: {age}, Gender: {gender}"
        detections.append(label)

        # Create a temporary file for each iteration
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
            temp_filename = temp_file.name
            temp_files.append(temp_filename)

        # Convert text to speech and play it
        text_to_speech(label, temp_filename)

        # Use putText method for inserting text on video
        cv2.putText(frame, label, (50, 50), font, 1, (0, 0, 255), 2, cv2.LINE_4)
        cv2.imshow("Original video", frame)

        if cv2.waitKey(2) & 0xFF == ord("q"):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
    delete_mp3_files(temp_files)
