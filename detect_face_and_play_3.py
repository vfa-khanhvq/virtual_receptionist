import cv2
import time
import boto3
from gtts import gTTS
from io import BytesIO
from pydub import AudioSegment
from pydub.playback import play

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

# Function to detect and identify faces using XML file
def detect_and_identify_faces(frame):
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=2, minNeighbors=5)
        
        face_bounding_boxes = []
        names = []

        for (x, y, w, h) in faces:
            # Save bounding box coordinates
            face_bounding_boxes.append((x, y, w, h))

            # Extract the face image
            face_image = frame[y:y+h, x:x+w]
            face_image_bytes = image_to_bytes(face_image)

            # Identify the face using the extracted image
            response_search = client.search_faces_by_image(
                CollectionId=collection_id,
                Image={'Bytes': face_image_bytes},
                MaxFaces=1,
                FaceMatchThreshold=80
            )

            if response_search['FaceMatches']:
                name = response_search['FaceMatches'][0]['Face']['ExternalImageId'] 
                if name not in announced_names:
                    names.append(name)
            else:
                names.append('Unknown')
                # Save the frame as a jpg file
                filename = f'images/face_{time.strftime("%Y%m%d_%H%M%S")}.jpg'
                cv2.imwrite(filename, face_image)
            
            # Perform face recognition or identification here if needed
            # For simplicity, we'll just append 'Unknown' for all detected faces
            # names.append('Unknown')

        return names, face_bounding_boxes
    except Exception as e:
        print(f"Error detecting and identifying faces: {e}")
        return [], []

# Initialize the webcam
cap = cv2.VideoCapture(0)  # Adjust camera index as needed
last_detection_time = time.time()

# Initialize names and bounding_boxes to empty lists
names = []
bounding_boxes = []
announced_names = set()
user_data = {
    'khanhvq': 'Khánh Vũ',
    'huutc': 'Hữu Michael',
    'khanhld': 'Khánh LD',
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
    if current_time - last_detection_time >= 1:
        names, bounding_boxes = detect_and_identify_faces(frame)
        print(names)
        last_detection_time = current_time

    # Display the results
    for i, (name, box) in enumerate(zip(names, bounding_boxes)):
        x, y, w, h = box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    # Play sound names
    if len(names) > 0:
        if 'Unknown' in names:
            str_name = 'Chào mừng bạn đến với SPL lab.'
        else:
            str_name = []
            for name in names:
                str_name.append(user_data.get(name, name))
                announced_names.add(name)
            str_name = 'và '.join(str_name)
            str_name = 'Xin chào! ' + str_name + '.'

        # Convert text to speech
        tts = gTTS(text=str_name, lang='vi')
        tts_fp = BytesIO()
        tts.write_to_fp(tts_fp)
        tts_fp.seek(0)
        
        try:
            # Load the audio into an AudioSegment
            audio = AudioSegment.from_file(tts_fp, format="mp3")
            # Play the audio
            play(audio)
        except Exception as e:
            print(f"Error playing audio: {e}")

    # Display the frame
    # cv2.imshow('Video', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
