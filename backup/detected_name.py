import cv2
import boto3

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
        response = client.search_faces_by_image(
            CollectionId=collection_id,
            Image={'Bytes': image_bytes},
            MaxFaces=1,
            FaceMatchThreshold=95
        )

        if 'FaceMatches' in response and response['FaceMatches']:
            names = [match['Face']['ExternalImageId'] for match in response['FaceMatches']]
        else:
            names = []
        
    except client.exceptions.InvalidParameterException as e:
        print(f"No faces detected in the frame: {e}")
        names = []

    return names

# Initialize the webcam
cap = cv2.VideoCapture(1)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    # Detect and identify faces
    names = detect_and_identify_faces(frame)

    # Display the results
    if names:
        for i, name in enumerate(names):
            cv2.putText(frame, name, (50, 50 + i * 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, "no people", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow('Video', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
