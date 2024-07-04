import cv2
import boto3
import numpy as np

# Initialize the Rekognition client
rekognition = boto3.client('rekognition', region_name='us-east-1')  # Replace with your desired AWS region

# Function to process frames from webcam
def process_frames():
    cap = cv2.VideoCapture(1)  # Open the webcam (0 is usually the default webcam index)

    while True:
        ret, frame = cap.read()  # Read frame from webcam

        if not ret:
            print("Failed to capture image from webcam.")
            break

        # Convert OpenCV BGR format to RGB (which is required by Rekognition)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Prepare the image for Amazon Rekognition (convert to bytes)
        _, img_encoded = cv2.imencode('.jpg', frame_rgb)
        img_bytes = img_encoded.tobytes()

        # Call Amazon Rekognition to detect faces in the image
        response = rekognition.detect_faces(
            Image={
                'Bytes': img_bytes
            },
            Attributes=['ALL']  # Optional: specify additional attributes like 'DEFAULT' or 'ALL'
        )

        # Process the response
        if 'FaceDetails' in response:
            for face_detail in response['FaceDetails']:
                # Extract bounding box coordinates
                box = face_detail['BoundingBox']
                height, width, _ = frame.shape
                left = int(box['Left'] * width)
                top = int(box['Top'] * height)
                right = int((box['Left'] + box['Width']) * width)
                bottom = int((box['Top'] + box['Height']) * height)

                # Draw bounding box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)

        # Display the frame with bounding boxes
        cv2.imshow('Amazon Rekognition Face Detection', frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture and close all windows
    cap.release()
    cv2.destroyAllWindows()

# Run the function to start processing frames from webcam
process_frames()
