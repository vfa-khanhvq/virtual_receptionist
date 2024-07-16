import boto3

# Initialize the boto3 client
client = boto3.client('rekognition')

# Specify the collection ID
collection_id = 'my_face_collection'

# List faces in the collection
response = client.list_faces(CollectionId=collection_id)

# Extract and print all unique names (external image IDs)
names = {}

while True:
    faces = response['Faces']
    
    for face in faces:
        print(face)
        name = face['ExternalImageId']
        face = face['FaceId']
        if name not in names:
            names[name] = []
        names[name].append(face)
    
    # Check if there are more faces to retrieve
    if 'NextToken' in response:
        next_token = response['NextToken']
        response = client.list_faces(CollectionId=collection_id, NextToken=next_token)
    else:
        break

print("Names in the collection:")
for name in names:
    print(name)
    print(names[name])
    response = client.associate_faces(
            CollectionId=collection_id,
            UserId=name,
            FaceIds=names[name]
        )
