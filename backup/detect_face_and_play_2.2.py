import cv2
import boto3
import time
from gtts import gTTS
from io import BytesIO
from pydub import AudioSegment
from pydub.playback import play
import random
import threading

# Initialize the boto3 client
client = boto3.client('rekognition')

# Load the pre-trained Haar Cascade file for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Specify the collection ID
collection_id = 'my_face_collection'

# Function to convert image to bytes
def image_to_bytes(image):
    _, buffer = cv2.imencode('.jpg', image)
    return buffer.tobytes()

announced_names = set()
user_data = {
    'khanhvq': 'Khánh Vũ',
    'huutc': 'Hữu Michael',
    'khanhld': 'Khánh LD',
}
hello_text = [
    'Xin chào, {} .',
    'Ye ye, vào thôi {} ơi',
    'Vào nhanh nào {} ',
    'Đến muộn rồi nha {} ',
    'Vào nhanh không bị phạt {} ơi ',
    'Lêu lêu {} ',
]

# Function to detect and identify faces
def detect_and_identify_faces(frame):
    image_bytes = image_to_bytes(frame)
    names_and_bounding_boxes = {}
    try:
        # Check if the frame has a face before searching names to save on API calls
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3)
        if len(faces) == 0:
            return names_and_bounding_boxes

        # Identify the face using the extracted image
        print('Requesting AWS Rekognition to search for face...')
        response_search = client.search_faces_by_image(
            CollectionId=collection_id,
            Image={'Bytes': image_bytes},
            MaxFaces=5,
            FaceMatchThreshold=80
        )

        if response_search['FaceMatches']:
            for face_match in response_search['FaceMatches']:
                face = face_match['Face']
                name = face['ExternalImageId']
                if name not in announced_names:
                    bounding_box = face['BoundingBox']
                    names_and_bounding_boxes[name] = bounding_box
        else:
            names_and_bounding_boxes['Unknown'] = {}
            # Save the frame as a jpg file
            filename = f'images/face_{time.strftime("%Y%m%d_%H%M%S")}.jpg'
            cv2.imwrite(filename, frame)

        return names_and_bounding_boxes

    except Exception as e:
        print(f"Error detecting and identifying faces: {e}")
        return names_and_bounding_boxes

# Function to detect emotions
def detect_emotions(image_bytes, bounding_boxes):
    try:
        print('Requesting AWS Rekognition to detect emotions...')
        response_detect = client.detect_faces(
            Image={'Bytes': image_bytes},
            Attributes=['ALL']
        )
        
        emotions = {}
        for face_detail in response_detect['FaceDetails']:
            emotions_list = face_detail['Emotions']
            # Get the emotion with the highest confidence
            primary_emotion = max(emotions_list, key=lambda item: item['Confidence'])
            print(emotions_list)
            happiness = next((item['Confidence'] for item in emotions_list if item['Type'] == 'HAPPY'), 0)
            print(happiness)
            bounding_box = face_detail['BoundingBox']
            # Find the corresponding bounding box index
            for i, box in enumerate(bounding_boxes.values()):
                if box == bounding_box:
                    emotions[i] = {'type': primary_emotion['Type'], 'happiness': happiness}
                    break
        
        return emotions

    except Exception as e:
        print(f"Error detecting emotions: {e}")
        return {}

# Function to play welcome message
def play_welcome_message(message):
    tts = gTTS(text=message, lang='vi')
    tts_fp = BytesIO()
    tts.write_to_fp(tts_fp)
    tts_fp.seek(0)
    audio = AudioSegment.from_file(tts_fp, format="mp3")
    play(audio)

# Initialize the webcam
cap = cv2.VideoCapture(0)
last_detection_time = time.time()
last_play_welcome = time.time()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    # Get the current time
    current_time = time.time()

    # Call detect_and_identify_faces function once every second
    if current_time - last_detection_time >= 10:
        names_and_bounding_boxes = detect_and_identify_faces(frame)
        last_detection_time = current_time

        # Play sound names
        if names_and_bounding_boxes:
            str_message = ''
            str_name = ''
            if 'Unknown' in names_and_bounding_boxes and current_time - last_play_welcome >= 10:
                str_message = 'Chào mừng bạn đến với SPL.'
                last_play_welcome = current_time
            else:
                real_names = []
                for name in names_and_bounding_boxes:
                    if name != 'Unknown':
                        real_names.append(user_data.get(name, name))
                        # announced_names.add(name)
                if real_names:
                    str_name = 'và '.join(real_names)
                    random_hello_text = random.choice(hello_text)
                    str_message = random_hello_text.format(str_name)
            if str_message:
                threading.Thread(target=play_welcome_message, args=(str_message,)).start()

        # Detect and display emotions after playing the welcome message
        if names_and_bounding_boxes:
            image_bytes = image_to_bytes(frame)
            emotions = detect_emotions(image_bytes, names_and_bounding_boxes)
            for i, (name, bbox) in enumerate(names_and_bounding_boxes.items()):
                x, y, w, h = int(bbox['Left']*frame.shape[1]), int(bbox['Top']*frame.shape[0]), int(bbox['Width']*frame.shape[1]), int(bbox['Height']*frame.shape[0])
                emotion_data = emotions.get(i, {'type': 'Unknown', 'happiness': 0})
                emotion_type = emotion_data['type']
                happiness = emotion_data['happiness']
                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # Display name and emotion
                display_text = f"{name}: {emotion_type} ({happiness:.2f}%)"
                cv2.putText(frame, display_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        
    # Display the frame
    cv2.imshow('Video', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
