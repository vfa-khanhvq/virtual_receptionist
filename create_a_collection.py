import boto3

# Initialize the boto3 client
client = boto3.client('rekognition')

# Create a collection
collection_id = 'my_face_collection'
response = client.create_collection(CollectionId=collection_id)

print('Collection ARN: ' + response['CollectionArn'])
print('Status code: ' + str(response['StatusCode']))
