import boto3
import os

# Initialize the boto3 client
client = boto3.client('rekognition')

# Specify the collection ID
collection_id = 'my_face_collection'

# Path to the root directory containing person directories
root_directory = 'dataset'

# Iterate over each person directory
for person_name in os.listdir(root_directory):
    person_directory = os.path.join(root_directory, person_name)
    print(person_directory)
    if os.path.isdir(person_directory):
        # Iterate over images in the person's directory
        for image_name in os.listdir(person_directory):
            if image_name.lower().endswith(('png', 'jpg', 'jpeg')):
                image_path = os.path.join(person_directory, image_name)
                
                # Read image bytes
                with open(image_path, 'rb') as image_file:
                    image_bytes = image_file.read()
                
                # Index the face
                response = client.index_faces(
                    CollectionId=collection_id,
                    Image={'Bytes': image_bytes},
                    ExternalImageId=person_name,  # Use directory name as the external ID
                    MaxFaces=1,
                    QualityFilter="AUTO",
                    DetectionAttributes=['ALL']
                )
                
                print(f'Indexed face for {image_name} under {person_name}')
                for faceRecord in response['FaceRecords']:
                    print('Face ID: ' + faceRecord['Face']['FaceId'])
                    print('Location: {}'.format(faceRecord['Face']['BoundingBox']))
