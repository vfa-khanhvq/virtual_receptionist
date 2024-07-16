import cv2
import face_recognition
import time
from gtts import gTTS
from io import BytesIO
from pydub import AudioSegment
from pydub.playback import play
import random
import threading

# Load the pre-trained Haar Cascade file for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load known faces and names
known_face_encodings = []
known_face_names = []

user_data = {
    'khanhvq': 'Khánh Vũ',
    'huutc': 'Hữu Michael',
    'khanhld': 'Khánh LD',
}

# Load user images and encode faces
for user, name in user_data.items():
    user_image = face_recognition.load_image_file(f'path_to_images/{user}.jpg')
    user_face_encoding = face_recognition.face_encodings(user_image)[0]
    known_face_encodings.append(user_face_encoding)
    known_face_names.append(user)

hello_text = [
    'Xin chào, {} .',
    'Ye ye, vào thôi {} ơi',
    'Vào nhanh nào {} ',
    'Đến muộn rồi nha {} ',
    'Vào nhanh không bị phạt {} ơi ',
    'Lêu lêu {} ',
]

announced_names = set()

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

    # Resize frame for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Get the current time
    current_time = time.time()

    # Call detect_and_identify_faces function once every second
    if current_time - last_detection_time >= 1:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        names = {}
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # If a match was found in known_face_encodings, use the first one
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

            if name not in announced_names:
                names[name] = None  # Emotions detection can be added here if needed
            else:
                names['Unknown'] = None
                filename = f'images/face_{time.strftime("%Y%m%d_%H%M%S")}.jpg'
                cv2.imwrite(filename, frame)

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
                        real_names.append(user_data.get(name))
                    announced_names.add(name)

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
