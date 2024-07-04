import cv2
import numpy as np
import os
from gtts import gTTS
from playsound import playsound
import time

# Load the pre-trained model
modelFile = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
configFile = "deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)

# Function to announce names using gTTS
announced_names = set()
def announce_names(names):
    global announced_names
    for name in names:
        if name != 'Unknown' and name not in announced_names:
            tts = gTTS(text=name, lang='en')
            filename = f'{name}.mp3'
            tts.save(filename)
            playsound(filename)
            announced_names.add(name)
            os.remove(filename)

# Initialize webcam
cap = cv2.VideoCapture(1)
last_detection_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    current_time = time.time()

    if current_time - last_detection_time >= 1:
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()

        names = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Dummy names for demonstration
                name = "Person " + str(i + 1)
                names.append(name)
                cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        announce_names(names)
        last_detection_time = current_time

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
