import pickle

import numpy as np

import sys
from pathlib import Path

# Add the src directory to sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from classifier import NearestNeighborClassifier

# Class label for unknown subjects in test and training data.
UNKNOWN_LABEL = -1


# Evaluation of open-set face identification.
class OpenSetEvaluation:

    def __init__(
        self,
        classifier=NearestNeighborClassifier(),
        false_alarm_rate_range=np.logspace(-3, 0, 1000, endpoint=True),
    ):
        # The false alarm rates.
        self.false_alarm_rate_range = false_alarm_rate_range

        # Datasets (embeddings + labels) used for training and testing.
        self.train_embeddings = []
        self.train_labels = []
        self.test_embeddings = []
        self.test_labels = []

        # The evaluated classifier (see classifier.py)
        self.classifier = classifier

    # Prepare the evaluation by reading training and test data from file.
    def prepare_input_data(self, train_data_file, test_data_file):
        with open(train_data_file, "rb") as f:
            (self.train_embeddings, self.train_labels) = pickle.load(f, encoding="bytes")
        with open(test_data_file, "rb") as f:
            (self.test_embeddings, self.test_labels) = pickle.load(f, encoding="bytes")

    # Run the evaluation and find performance measure (identification rates) at different
    # similarity thresholds.
    def run(self):
        # Step 1: Train the classifier
        self.classifier.fit(self.train_embeddings, self.train_labels)

        # Step 2: Predict labels and similarities for the test set
        prediction_labels, similarities = self.classifier.predict_labels_and_similarities(self.test_embeddings)

        # Step 3: Initialize lists for storing results
        similarity_thresholds = []
        identification_rates = []

        # Step 4: Loop over each FAR
        for far in self.false_alarm_rate_range:
            # Compute the similarity threshold
            threshold = self.select_similarity_threshold(similarities, far)
            similarity_thresholds.append(threshold)

            # Compute identification rate for the current threshold
            identified_labels = [
                label if sim >= threshold else UNKNOWN_LABEL
                for label, sim in zip(prediction_labels, similarities)
            ]
            identification_rate = self.calc_identification_rate(identified_labels)
            identification_rates.append(identification_rate)

        # Return the results
        evaluation_results = {
            "similarity_thresholds": similarity_thresholds,
            "identification_rates": identification_rates,
        }
        return evaluation_results

        # similarity_thresholds = None
        # identification_rates = None
        #
        # # Report all performance measures.
        # evaluation_results = {
        #     "similarity_thresholds": similarity_thresholds,
        #     "identification_rates": identification_rates,
        # }
        #
        # return evaluation_results

    def select_similarity_threshold(self, similarity, false_alarm_rate):
        # Calculate the p-percentile based on the given FAR
        percentile = 100 * (1 - false_alarm_rate)

        # Find the similarity threshold using np.percentile
        similarity_threshold = np.percentile(similarity, percentile)
        return similarity_threshold

    def calc_identification_rate(self, prediction_labels):
        # Convert to NumPy arrays for easier operations
        prediction_labels = np.array(prediction_labels)
        true_labels = np.array(self.test_labels)

        # Count correctly predicted labels
        correct_predictions = np.sum(prediction_labels == true_labels)

        # Compute identification rate
        identification_rate = correct_predictions / len(true_labels)
        return identification_rate
