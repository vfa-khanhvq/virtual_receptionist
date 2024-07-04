import os
import numpy as np
from sklearn import neighbors
import pickle

def train_model(dataset_path, model_save_path):
    X = []
    y = []
    for person_name in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person_name)
        if not os.path.isdir(person_path):
            continue
        for image_name in os.listdir(person_path):
            image_path = os.path.join(person_path, image_name)
            image = face_recognition.load_image_file(image_path)
            encoding = get_face_encodings(image)
            if encoding is not None:
                X.append(encoding)
                y.append(person_name)

    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=2, algorithm='ball_tree', weights='distance')
    knn_clf.fit(X, y)

    with open(model_save_path, 'wb') as f:
        pickle.dump(knn_clf, f)

train_model("dataset", "knn_model.pkl")
