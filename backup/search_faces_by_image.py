import cv2
import boto3
from PIL import Image
import io
from gtts import gTTS
import pygame
import tempfile
import time

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
}
# Initialize the webcam
# cap = cv2.VideoCapture(1)
video_path = '02.mp4'
cap = cv2.VideoCapture(video_path)
collection_id = "my_face_collection"
last_play_welcome = time.time()
current_time = time.time()
def load_image(frame):
    _, buffer = cv2.imencode('.jpg', frame)
    return buffer.tobytes()

def detect_faces(frame):
    # print(f'Detecting faces in image: {frame}')
    response = client.detect_faces(
        Image=load_image(frame),
        Attributes=['ALL']
    )
    face_details = response['FaceDetails']
    print(f'- found {len(face_details)} faces')

    faces = []
    for face in face_details:
        bounding_box = face['BoundingBox']
        happiness = next((emotion['Confidence'] for emotion in face['Emotions'] if emotion['Type'] == 'HAPPY'), 0)
        faces.append((bounding_box, happiness))
    return faces

def crop_face(frame, bounding_box):
    image = Image.open(frame)
    width, height = image.size

    left = int(bounding_box['Left'] * width)
    top = int(bounding_box['Top'] * height)
    right = left + int(bounding_box['Width'] * width)
    bottom = top + int(bounding_box['Height'] * height)

    face_image = image.crop((left, top, right, bottom))
    return face_image

def search_users_by_image(collection_id, frame):
    faces = detect_faces(frame)
    if len(faces) == 0:
        print('No faces detected. Exiting.')
        return {"UserMatches": [], "UserNotMatches": []}

    user_matches = []
    user_not_matches = []

    for i, (bounding_box, happiness) in enumerate(faces):
        cropped_face = crop_face(frame, bounding_box)
        
        # Convert cropped face to bytes
        with io.BytesIO() as output:
            cropped_face.save(output, format="JPEG")
            cropped_face_bytes = output.getvalue()

        response = client.search_users_by_image(
            CollectionId=collection_id,
            Image={'Bytes': cropped_face_bytes},
            MaxUsers=5,
            UserMatchThreshold=80
        )

        if "UserMatches" in response and response["UserMatches"]:
            for match in response["UserMatches"]:
                user_id = match["User"].get("UserId", f'user{i+1}')
                similarity = match["Similarity"]
                if user_id in user_data:
                    real_name = user_data[user_id]
                    user_matches.append({"name": real_name, "happiness": happiness, "similarity": similarity})
                    print(f'- {real_name} - {similarity}% Happiness: {happiness:.2f}%')
                else:
                    user_matches.append({"name": user_id, "happiness": happiness, "similarity": similarity})
                    print(f'- {user_id} - {similarity}% Happiness: {happiness:.2f}%')
        else:
            user_id = f'user{i+1}'
            user_not_matches.append({"name": user_id, "happiness": happiness})
            print(f'- {user_id} - No match found. Happiness: {happiness:.2f}%')

    return {"UserMatches": user_matches, "UserNotMatches": user_not_matches}

def generate_and_play_greeting(user_matches, user_not_matches):
    global current_time
    global last_play_welcome
    greeting_message = ""
    print(current_time)
    print(last_play_welcome)
    if user_not_matches and current_time - last_play_welcome >= 10:
        greeting_message += "Chào mừng bạn đến với SPL. "
        last_play_welcome = time.time()
    if user_matches:
        names = ' và '.join([user["name"] for user in user_matches])
        greeting_message += f"Xin chào {names}."
    
    if greeting_message:
        tts = gTTS(greeting_message, lang='vi')
        with tempfile.NamedTemporaryFile(delete=True) as fp:
            tts.save(fp.name)
            fp.seek(0)
            pygame.mixer.init()
            pygame.mixer.music.load(fp.name)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                continue

# def main():
#     collection_id = "my_face_collection"
#     IMAGE_SEARCH_SOURCE = '05.JPG'
#     results = search_users_by_image(collection_id, IMAGE_SEARCH_SOURCE)
    
#     # Format and print the results
#     user_matches = results['UserMatches']
#     user_not_matches = results['UserNotMatches']
    
#     if user_matches:
#         greetings = ', '.join([f'{user["name"]} ({user["happiness"]:.2f}%)' for user in user_matches])
#         print(f'Xin chào: {greetings}')
    
#     if user_not_matches:
#         welcomes = ', '.join([f'{user["name"]} ({user["happiness"]:.2f}%)' for user in user_not_matches])
#         print(f'Chào mừng đến với SPL: {welcomes}')
    
#     # Generate and play greeting message
#     generate_and_play_greeting(user_matches, user_not_matches)

def main():
    global current_time
    # Initialize the face detection model
    while True:
        # Get the current time
        current_time = time.time()
        # Capture frame-by-frame
        ret, frame = cap.read()
        results = search_users_by_image(collection_id, frame)
        print(results)
        if not ret:
            print("Failed to grab frame")
            break

        
        # Display the frame
        cv2.imshow('Video', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
if __name__ == "__main__":
    main()
