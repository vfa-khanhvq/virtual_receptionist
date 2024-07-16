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
    # Index the face
    response = client.create_user(
        CollectionId=collection_id,
        UserId=person_name,
    )
                
    print(f"User '{person_name}' added to collection '{collection_id}'.")
