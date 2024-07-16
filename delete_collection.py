import boto3

def delete_rekognition_collection(collection_id):
    # Initialize the boto3 client
    client = boto3.client('rekognition')
    
    try:
        # Attempt to delete the collection
        response = client.delete_collection(CollectionId=collection_id)
        
        # Check if the collection was deleted successfully
        status_code = response['StatusCode']
        if status_code == 200:
            print(f"Collection '{collection_id}' deleted successfully.")
        else:
            print(f"Failed to delete collection '{collection_id}'. Status code: {status_code}")
    
    except client.exceptions.ResourceNotFoundException:
        print(f"Collection '{collection_id}' does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")

# Specify the collection ID you want to delete
collection_id = 'my_face_collection'

# Call the function to delete the collection
delete_rekognition_collection(collection_id)
