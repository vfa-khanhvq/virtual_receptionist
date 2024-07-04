import cv2
import boto3
import time
from gtts import gTTS
from io import BytesIO
from pydub import AudioSegment
from pydub.playback import play

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
            MaxFaces=10,  # Adjust as needed to limit the number of faces
            FaceMatchThreshold=80  # Adjust as needed
        )

        names = set()  # Use a set to store names to automatically handle duplicates
        face_matches = response['FaceMatches']

        for match in face_matches:
            face = match['Face']
            names.add(face['ExternalImageId'])

        if not names:
            names.add('Unknown')

        return list(names)  # Convert the set back to a list

    except Exception as e:
        print(f"Error detecting and identifying faces: {e}")
        return []

# Initialize the webcam
cap = cv2.VideoCapture(1)
last_detection_time = time.time()

# Initialize names to an empty list
names = []
announced_names = set()
user_data = {
    'khanhvq': 'Khánh Vũ',
    'huutc': 'Hữu Michael',
}

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    # Get the current time
    current_time = time.time()

    # Call detect_and_identify_faces function once every second
    if current_time - last_detection_time >= 2:
        names = detect_and_identify_faces(frame)
        last_detection_time = current_time

    # Display the results
    for i, name in enumerate(names):
        if name != 'Unknown' and name not in announced_names:
            # Play sound name
            # Vietnamese text
            text = "Xin chào! " + user_data[name] + '.'

            # Convert text to speech
            tts = gTTS(text=text, lang='vi')
            tts_fp = BytesIO()
            tts.write_to_fp(tts_fp)
            tts_fp.seek(0)

            # Load the audio into an AudioSegment
            audio = AudioSegment.from_file(tts_fp, format="mp3")

            # Play the audio
            play(audio)
            announced_names.add(name)

    # Display the frame
    # cv2.imshow('Video', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
