import cv2
import boto3
import time

# Initialize the boto3 client
client = boto3.client('rekognition')

# Specify the collection ID
collection_id = 'my_face_collection'

# Function to convert image to bytes
def image_to_bytes(image):
    _, buffer = cv2.imencode('.jpg', image)
    return buffer.tobytes()

# Function to detect and identify faces
def detect_and_identify_faces(frame):
    image_bytes = image_to_bytes(frame)

    try:
        # First detect faces to get bounding boxes
        response_detect = client.detect_faces(
            Image={'Bytes': image_bytes},
            Attributes=['ALL']
        )

        face_details = response_detect['FaceDetails']

        face_bounding_boxes = []
        names = []

        for face_detail in face_details:
            box = face_detail['BoundingBox']
            left = int(frame.shape[1] * box['Left'])
            top = int(frame.shape[0] * box['Top'])
            width = int(frame.shape[1] * box['Width'])
            height = int(frame.shape[0] * box['Height'])

            # Save bounding box coordinates
            face_bounding_boxes.append((left, top, width, height))

            # Extract the face image
            face_image = frame[top:top+height, left:left+width]
            face_image_bytes = image_to_bytes(face_image)

            # Identify the face using the extracted image
            response_search = client.search_faces_by_image(
                CollectionId=collection_id,
                Image={'Bytes': face_image_bytes},
                MaxFaces=1,
                FaceMatchThreshold=80
            )

            if response_search['FaceMatches']:
                names.append(response_search['FaceMatches'][0]['Face']['ExternalImageId'])
            else:
                names.append('Unknown')

        if not names:
            names.append('Unknown')

        return names, face_bounding_boxes

    except Exception as e:
        print(f"Error detecting and identifying faces: {e}")
        return [], []

# Initialize the webcam
cap = cv2.VideoCapture(1)
last_detection_time = time.time()

# Initialize names and bounding_boxes to empty lists
names = []
bounding_boxes = []

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    # Get the current time
    current_time = time.time()

    # Call detect_and_identify_faces function once every second
    if current_time - last_detection_time >= 1:
        names, bounding_boxes = detect_and_identify_faces(frame)
        print(names)
        last_detection_time = current_time

    # Display the results
    for i, (name, box) in enumerate(zip(names, bounding_boxes)):
        left, top, width, height = box
        # cv2.rectangle(frame, (left, top), (left + width, top + height), (255, 0, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # Display the frame
    cv2.imshow('Video', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
