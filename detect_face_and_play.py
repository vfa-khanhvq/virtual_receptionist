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

bounding_boxes = []
announced_names = set()
user_data = {
    'khanhvq': 'Khánh Vũ',
    'huutc': 'Hữu Michael',
    'khanhld': 'Khánh LD',
    'duyna': 'Duy',
    'tudl': 'Tú',
    'baopg': 'Bảo',
    'nguyennhp': 'Nguyên PHP',
    'thangnt': 'Thăng',
    'tuyenbq': 'Tuyến BQ',
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
    names = {}
    try:
        # Check frame have face or not before searching names to save costly API call
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3)
        if len(faces) == 0:
            return names
        # Identify the face using the extracted image
        print('Request AWS Rekognition search face...')
        response_search = client.search_faces_by_image(
            CollectionId=collection_id,
            Image={'Bytes': image_bytes},
            MaxFaces=5,
            FaceMatchThreshold=70
        )

        if response_search['FaceMatches']:
            faces = response_search['FaceMatches']
            print(faces)
            for match in faces:
                face = match['Face']
                name = face['ExternalImageId']
                print(name)
                # if name not in announced_names:
                names[name] = detect_emotions(frame, face['BoundingBox'])
        else:
            names['Unknown']= detect_emotions(frame)
            # Save the frame as a jpg file
            filename = f'images/face_{time.strftime("%Y%m%d_%H%M%S")}.jpg'
            cv2.imwrite(filename, frame)

        return names

    except Exception as e:
        print(f"Error detecting and identifying faces: {e}")
        return names


# Function to detect emotions
def detect_emotions(frame, box = None):
    face_image_bytes = image_to_bytes(frame)
    try:
        if box:
            left = int(frame.shape[1] * box['Left'])
            top = int(frame.shape[0] * box['Top'])
            width = int(frame.shape[1] * box['Width'])
            height = int(frame.shape[0] * box['Height'])

            # Extract the face image
            face_image = frame[top:top+height, left:left+width]
            face_image_bytes = image_to_bytes(face_image)

        print('Requesting AWS Rekognition to detect emotions...')
        response_detect = client.detect_faces(
            Image={'Bytes': face_image_bytes},
            Attributes=['ALL']
        )
        
        emotions = []
        for i, face_detail in enumerate(response_detect['FaceDetails']):
            emotions_list = face_detail['Emotions']
            # print(face_detail)
            # Get the emotion with the highest confidence
            emotion = next((item['Confidence'] for item in emotions_list if item['Type'] == 'HAPPY'), 0)
            emotions.append(emotion)
            # filename = f'images/face_emotion_{time.strftime("%Y%m%d_%H%M%S")}_{emotion}.jpg'
            # cv2.imwrite(filename, face_image)
            # print('Emotion: ', max(emotions))
            return max(emotions)
        return 0

    except Exception as e:
        print(f"Error detecting emotions: {e}")
        return 0

# Function to play welcome message
def play_welcome_message(message):
    tts = gTTS(text=message, lang='vi')
    tts_fp = BytesIO()
    tts.write_to_fp(tts_fp)
    tts_fp.seek(0)
    audio = AudioSegment.from_file(tts_fp, format="mp3")
    play(audio)

# Initialize the webcam
cap = cv2.VideoCapture(1)
# video_path = '02.mp4'
# cap = cv2.VideoCapture(video_path)
last_detection_time = time.time()
last_play_welcome = time.time()

# Initialize names and bounding_boxes to empty lists
names = []
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
        # print(names)
        last_detection_time = current_time

        # Play sound names
        if len(names) > 0:
            has_new = False
            str_message = ''
            str_name = ''
            str_print = ''
            if 'Unknown' in names and current_time - last_play_welcome >= 10:
                str_message = 'Chào mừng bạn đến với SPL.'
                last_play_welcome = current_time
            else:
                real_names = []
                for name in names:
                    if name != 'Unknown':
                        real_names.append(user_data.get(name,name))
                    announced_names.add(name)
                emotion = names[name]
                # print(emotion)
                str_print = str_print.join(f"{name}:{emotion} %")
                print(str_print)
                if len(real_names) > 0:
                    str_name = 'và '.join(real_names)
                    random_hello_text = random.choice(hello_text)
                    str_name = random_hello_text.format(str_name)
            str_message = str_message + str_name
            if str_message:
                cv2.putText(frame, str_print, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                threading.Thread(target=play_welcome_message, args=(str_message,)).start()
        
    # Display the frame
    cv2.imshow('Video', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
