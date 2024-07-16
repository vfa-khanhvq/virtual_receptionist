import cv2
import boto3
import time
from gtts import gTTS
from io import BytesIO
from pydub import AudioSegment
from pydub.playback import play
import random

# Initialize the boto3 client
client = boto3.client('rekognition')

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
    names = set()
    try:
        # Identify the face using the extracted image
        response_search = client.search_faces_by_image(
            CollectionId=collection_id,
            Image={'Bytes': image_bytes},
            MaxFaces=1,
            FaceMatchThreshold=80
        )

        if response_search['FaceMatches']:
            name = response_search['FaceMatches'][0]['Face']['ExternalImageId']
            if name not in announced_names:
                names.add(name)
        else:
            names.add('Unknown')
            # Save the frame as a jpg file
            filename = f'images/face_{time.strftime("%Y%m%d_%H%M%S")}.jpg'
            cv2.imwrite(filename, image_bytes)

        return names

    except Exception as e:
        print(f"Error detecting and identifying faces: {e}")
        return names

# Initialize the webcam
cap = cv2.VideoCapture(1)
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
    if current_time - last_detection_time >= 1:
        names = detect_and_identify_faces(frame)
        print(names)
        last_detection_time = current_time

        # Play sound names
        if len(names) > 0:
            str_name = []
            if 'Unknown' in names:
                str_name.append('Chào mừng bạn đến với SPL lab.')
            else:
                str_name = []
                for name in names:
                    if name != 'Unknown':
                        str_name.append(user_data.get(name, 'bạn'))
                    announced_names.add(name)
                str_name = 'và '.join(str_name)
                random_hello_text = random.choice(hello_text)
                str_name = random_hello_text.format(str_name)
            cv2.putText(frame, 'và '.join(names), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            # Convert text to speech
            tts = gTTS(text=str_name, lang='vi')
            tts_fp = BytesIO()
            tts.write_to_fp(tts_fp)
            tts_fp.seek(0)
            # Load the audio into an AudioSegment
            audio = AudioSegment.from_file(tts_fp, format="mp3")
            # Play the audio
            play(audio)
        
    # Display the frame
    cv2.imshow('Video', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
