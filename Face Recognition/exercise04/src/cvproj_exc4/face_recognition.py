import os
import pickle
import csv
import cv2
import numpy as np

from config import Config

# FaceNet to extract face embeddings.
class FaceNet:

    def __init__(self):
        self.facenet = cv2.dnn.readNetFromONNX(Config.resnet50)

    # Predict embedding from a given face image.
    def predict(self, face):
        # Normalize face image using mean subtraction.
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB) - (131.0912, 103.8827, 91.4953)

        # Forward pass through deep neural network. The input size should be 224 x 224.
        reshaped = np.moveaxis(face, 2, 0)
        reshaped = np.expand_dims(reshaped, axis=0)
        self.facenet.setInput(reshaped)
        embedding = np.squeeze(self.facenet.forward())
        return embedding / np.linalg.norm(embedding)

    @classmethod
    @property
    def get_embedding_dimensionality(cls):
        """Get dimensionality of the extracted embeddings."""
        return 128


# The FaceRecognizer model enables supervised face identification.
class FaceRecognizer:

    # Prepare FaceRecognizer; specify all parameters for face identification.
    def __init__(self, num_neighbours=5, max_distance=0.7, min_prob=0.8):
        # TODO: Prepare FaceNet and set all parameters for kNN.
        self.facenet = FaceNet()  # Initialize FaceNet for embedding extraction.
        self.num_neighbours = num_neighbours
        self.max_distance = max_distance
        self.min_prob = min_prob

        # The underlying gallery: class labels and embeddings.
        self.labels = []
        self.embeddings = np.empty((0, FaceNet.get_embedding_dimensionality))
        self.embeddings_grayscale = np.empty((0, FaceNet.get_embedding_dimensionality)) # To store the grayscal version 

        self._handle_version_compatibility()

        # Load face recognizer from pickle file if available.
        if os.path.exists(Config.rec_gallery):
            self.load()

    # Save the trained model as a pickle file.
    def save(self):
        with open(Config.rec_gallery, "wb") as f:
            pickle.dump((self.labels, self.embeddings, self.embeddings_grayscale), f)

    # Load trained model from a pickle file.
    def load(self):
        with open(Config.rec_gallery, "rb") as f:
            (self.labels, self.embeddings, self.embeddings_grayscale) = pickle.load(f)

    # TODO: Train face identification with a new face with labeled identity.
    def partial_fit(self, face, label):
        face_grayscale = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        # Again converting back to RGB bcz FaceNet expects 3 channels always
        face_grayscale = cv2.cvtColor(face_grayscale, cv2.COLOR_GRAY2BGR)

        embedding = self.facenet.predict(face)
        embedding_grayscale = self.facenet.predict(face_grayscale)

        self.embeddings = np.vstack([self.embeddings, embedding])
        self.embeddings_grayscale = np.vstack([self.embeddings_grayscale, embedding_grayscale])
        self.labels.append(label)


    # TODO: Predict the identity for a new face.
    def predict(self, face):
        face_grayscale = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face_grayscale = cv2.cvtColor(face_grayscale, cv2.COLOR_GRAY2BGR)

        embedding = self.facenet.predict(face)
        embedding_grayscale = self.facenet.predict(face_grayscale)

        distances = np.linalg.norm(self.embeddings - embedding, axis=1)
        distances_grayscale = np.linalg.norm(self.embeddings_grayscale - embedding_grayscale, axis=1)
        combined_distances = (distances + distances_grayscale) / 2

        nearest_indices = np.argsort(combined_distances)[:self.num_neighbours]

        # We select majority labels among all the labels
        nearest_labels = [self.labels[i] for i in nearest_indices]
        predicted_label = max(set(nearest_labels), key=nearest_labels.count)

        num_class_matches = nearest_labels.count(predicted_label)
        posterior_prob = num_class_matches / self.num_neighbours

        class_distances = [
            combined_distances[i] for i in nearest_indices if self.labels[i] == predicted_label
        ]
        class_distance = min(class_distances)

        if class_distance > self.max_distance or posterior_prob < self.min_prob:
            predicted_label = Config.UNKNOWN_LABEL

        return {"label": predicted_label, "posterior_prob": posterior_prob, "distance": class_distance}
    
    def _handle_version_compatibility(self):
        if os.path.exists(Config.rec_gallery):
            try:
                with open(Config.rec_gallery, "rb") as f:
                    data = pickle.load(f)
                    if not (isinstance(data, tuple) and len(data) == Config.NEW_VERSION_PICKLE_TUPLE_LENGTH):
                        os.remove(Config.rec_gallery)
            except Exception:
                os.remove(Config.rec_gallery)


# The FaceClustering class enables unsupervised clustering of face images according to their
# identity and re-identification.
class FaceClustering:

    # Prepare FaceClustering; specify all parameters of clustering algorithm.
    def __init__(self, num_clusters=2, max_iter=35):
        # TODO: Prepare FaceNet.
        self.facenet = FaceNet()

        # The underlying gallery: embeddings without class labels.
        self.embeddings = np.empty((0, FaceNet.get_embedding_dimensionality))

        # Number of cluster centers for k-means clustering.
        self.num_clusters = num_clusters
        # Cluster centers.
        self.cluster_center = np.empty((num_clusters, FaceNet.get_embedding_dimensionality))
        # Cluster index associated with the different samples.
        self.cluster_membership = []

        # Maximum number of iterations for k-means clustering.
        self.max_iter = max_iter

        # Load face clustering from pickle file if available.
        if os.path.exists(Config.cluster_gallery):
            self.load()

    # Save the trained model as a pickle file.
    def save(self):
        with open(Config.cluster_gallery, "wb") as f:
            pickle.dump(
                (self.embeddings, self.num_clusters, self.cluster_center, self.cluster_membership),
                f,
            )

    # Load trained model from a pickle file.
    def load(self):
        with open(Config.cluster_gallery, "rb") as f:
            (self.embeddings, self.num_clusters, self.cluster_center, self.cluster_membership) = (
                pickle.load(f)
            )

    # TODO
    def partial_fit(self, face):
        embedding = self.facenet.predict(face)
        self.embeddings = np.vstack([self.embeddings, embedding])


    # TODO
    def fit(self, random_seed=None):
        if random_seed is not None:
            np.random.seed(random_seed)
    
        random_indices = np.random.choice(len(self.embeddings), self.num_clusters, replace=False)
        self.cluster_center = self.embeddings[random_indices]
        inertia_arr = []

        for _ in range(self.max_iter):
            distances = np.linalg.norm(
                self.embeddings[:, np.newaxis] - self.cluster_center, axis=2
            )
            self.cluster_membership = np.argmin(distances, axis=1)

            inertia = np.sum(np.min(distances, axis=1) ** 2)
            inertia_arr.append(inertia)

            new_centers = np.array([
                self.embeddings[self.cluster_membership == i].mean(axis=0)
                for i in range(self.num_clusters)
            ])

            if np.allclose(self.cluster_center, new_centers):
                break
            self.cluster_center = new_centers

        with open(Config.clustering_experiments_result_file, "a", newline="") as f:
            writer = csv.writer(f)
            # Write header only if the file is empty
            if f.tell() == 0:
                writer.writerow(["Num Clusters", "Max Iter", "Inertia Values", "Random Seed"])
            writer.writerow([self.num_clusters, self.max_iter, inertia_arr, random_seed])

    # TODO
    def predict(self, face):
        embedding = self.facenet.predict(face)
        distances = np.linalg.norm(self.cluster_center - embedding, axis=1)
        best_cluster = np.argmin(distances)
        return {"cluster": best_cluster, "distances": distances}
