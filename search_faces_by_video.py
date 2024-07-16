import boto3
from PIL import Image
import io
from gtts import gTTS
import pygame
import tempfile
import cv2
import time
import random

# Initialize the boto3 client
client = boto3.client('rekognition')

# User data mapping usernames to real names
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
    'duynk': 'Duy Nguyễn',
    'linhlt': 'Linh',
    'thanghc': 'Thắng',
}
hello_text = [
    'Xin chào, {} .',
    'Ye ye, vào thôi {} ơi',
    'Vào nhanh nào {} ',
    'Đến muộn rồi nha {} ',
    'Vào nhanh không bị phạt {} ơi ',
    'Lêu lêu {} ',
]
announced_names = set()
collection_id = "my_face_collection"
# Open video capture (you can replace '0' with your video file path)
# cap = cv2.VideoCapture('02.mp4')
cap = cv2.VideoCapture(1)
# Get the current time
current_time = time.time()
last_play_welcome = time.time()
last_detection_time = time.time()

# Load the pre-trained Haar Cascade file for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def is_frontal_face(pose):
    # Check if the face is front-facing based on yaw, pitch, and roll values
    # return (abs(pose['Yaw']) <= yaw_threshold and
    #         abs(pose['Pitch']) <= pitch_threshold and
    #         abs(pose['Roll']) <= roll_threshold)
    return abs(pose['Roll']) > -50 and abs(pose['Roll']) <= 50

def detect_faces(image_bytes):
    response = client.detect_faces(
        Image={'Bytes': image_bytes},
        Attributes=['ALL'],
    )
    face_details = response['FaceDetails']
    faces = []
    for face in face_details:
        if face['Confidence'] >= 80 and is_frontal_face(face['Pose']):
            bounding_box = face['BoundingBox']
            happiness = next((emotion['Confidence'] for emotion in face['Emotions'] if emotion['Type'] == 'HAPPY'), 0)
            faces.append((bounding_box, happiness))
    return faces

def crop_face(image, bounding_box):
    width, height = image.size

    left = int(bounding_box['Left'] * width)
    top = int(bounding_box['Top'] * height)
    right = left + int(bounding_box['Width'] * width)
    bottom = top + int(bounding_box['Height'] * height)

    face_image = image.crop((left, top, right, bottom))
    return face_image

def save_image_bytes(image_bytes, file_path):
    with open(file_path, 'wb') as file:
        file.write(image_bytes)

def search_users_by_image(collection_id, image_bytes):
    faces = detect_faces(image_bytes)
    if not faces:
        return {"UserMatches": [], "UserNotMatches": []}

    user_matches = []
    user_not_matches = []

    for i, (bounding_box, happiness) in enumerate(faces):
        cropped_face = crop_face(Image.open(io.BytesIO(image_bytes)), bounding_box)

        # Convert cropped face to bytes
        with io.BytesIO() as output:
            cropped_face.save(output, format="JPEG")
            cropped_face_bytes = output.getvalue()

        try:
            response = client.search_faces_by_image(
                CollectionId=collection_id,
                Image={'Bytes': cropped_face_bytes},
                MaxFaces=5,
                FaceMatchThreshold=70
            )

            if "FaceMatches" in response and response["FaceMatches"]:
                for match in response["FaceMatches"]:
                    external_image_id = match["Face"]["ExternalImageId"]
                    similarity = match["Similarity"]
                    if external_image_id in user_data:
                        real_name = user_data[external_image_id]
                        # Append only if the real_name is not already in user_matches
                        if real_name not in [user["name"] for user in user_matches]:
                            if(real_name not in announced_names):
                                user_matches.append({"name": real_name, "happiness": happiness, "similarity": similarity})
                                announced_names.add(real_name)
                                print(f'- {real_name} - {similarity}% Happiness: {happiness:.2f}%')
            else:
                user_id = f'user{i+1}'
                user_not_matches.append({"name": user_id, "happiness": happiness})
                filename = f'images/face_{time.strftime("%Y%m%d_%H%M%S")}.JPEG'
                save_image_bytes(cropped_face_bytes, filename)
                print(f'- {user_id} - No match found. Happiness: {happiness:.2f}%')
        except Exception as e:
            print(f'Error searching users by image: {e}')

    return {"UserMatches": user_matches, "UserNotMatches": user_not_matches}

def generate_and_play_greeting(user_matches, user_not_matches):
    global current_time
    global last_play_welcome
    greeting_message = ""
    if user_not_matches and current_time - last_play_welcome >= 10:
        greeting_message += "Chào mừng bạn đến với SPL. "
        last_play_welcome = current_time
    if user_matches:
        # Extract unique names from user_matches
        unique_names = []
        for user in user_matches:
            if user["name"] not in unique_names:
                unique_names.append(user["name"])

        names = ', '.join(unique_names)
        # greeting_message += f"Xin chào {names}."
        random_hello_text = random.choice(hello_text)
        greeting_message = random_hello_text.format(names)
    if greeting_message:
        try:
            tts = gTTS(greeting_message, lang='vi')
            with tempfile.NamedTemporaryFile(delete=True) as fp:
                tts.save(fp.name)
                fp.seek(0)
                pygame.mixer.init()
                pygame.mixer.music.load(fp.name)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    continue
        except Exception as e:
            print(f'Error generating or playing greeting message: {e}')

def main():
    global current_time
    global last_detection_time
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture image from video feed. Exiting...")
            break
        # Get the current time
        current_time = time.time()
        if current_time - last_detection_time >= 1:
            # Detect faces locally
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3)
            if len(faces) > 0:
                # Convert frame to bytes
                ret, buffer = cv2.imencode('.jpg', frame)
                image_bytes = buffer.tobytes()

                # Perform face recognition
                results = search_users_by_image(collection_id, image_bytes)
                user_matches = results['UserMatches']
                user_not_matches = results['UserNotMatches']

                # Generate and play greeting message
                generate_and_play_greeting(user_matches, user_not_matches)
            else:
                print('No faces detected')
        # Exit loop on key press 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
