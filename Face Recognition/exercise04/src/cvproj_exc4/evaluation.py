import pickle

import numpy as np

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
    ''' Returns an object with following structure:
        @identification_rates: [all identification rates],
        @best_sim_thresholds: 
            { 
                min_far_threshold: { value: value for threshold, index: index of value in FAR array }
                max_id_rate_threshold: { value: value for threshold, index: index of value in FAR array }
            }
        '''
    def run(self):
        self.classifier.fit(self.train_embeddings, self.train_labels)
        prediction_labels, similarities = self.classifier.predict_labels_and_similarities(self.test_embeddings)

        similarity_thresholds = []
        identification_rates = []

        for far in self.false_alarm_rate_range:
            threshold = self.select_similarity_threshold(similarities, far)
            similarity_thresholds.append(threshold)

            identified_labels = [
                label if sim >= threshold else UNKNOWN_LABEL
                for label, sim in zip(prediction_labels, similarities)
            ]
            identification_rate = self.calc_identification_rate(identified_labels)
            identification_rates.append(identification_rate)

        best_sim_thresholds = self.calculate_best_similarity_thresholds(identification_rates, similarity_thresholds)

        return {
            "identification_rates": identification_rates,
            "best_sim_thresholds": best_sim_thresholds
        }

    def select_similarity_threshold(self, similarity, false_alarm_rate):
        percentile = 100 * (1 - false_alarm_rate)
        return np.percentile(similarity, percentile)

    def calc_identification_rate(self, prediction_labels):
        prediction_labels = np.array(prediction_labels)
        true_labels = np.array(self.test_labels)
        ''' Since this is a OpenSet evaluation, there are Unknown labels in the dataset and we
        need to filter the OUT !!!!!!!'''
        known_mask = (true_labels != UNKNOWN_LABEL)
        correct_predictions = np.sum((prediction_labels[known_mask] == true_labels[known_mask]))
        return (correct_predictions / np.sum(known_mask))
    
    def calculate_best_similarity_thresholds(self, identification_rates, similarity_thresholds):
        indices_of_far_lteq_1p = np.where(self.false_alarm_rate_range <= 0.01)[0]
        max_iden_rate_index_for_far_lt_eq_1percent = indices_of_far_lteq_1p[np.argmax(
            [identification_rates[i] for i in indices_of_far_lteq_1p]
        )]
        best_sim_threshold_far_lteq_1p = similarity_thresholds[max_iden_rate_index_for_far_lt_eq_1percent]

        # Find the best threshold for Identification Rate ≥ 90% (minimize FAR)
        indices_of_id_rate_gteq_90p = np.where(np.array(identification_rates) >= 0.9)[0]
        min_far_index_id_rate_gteq_90 = indices_of_id_rate_gteq_90p[np.argmin(
            [self.false_alarm_rate_range[i] for i in indices_of_id_rate_gteq_90p]
        )]
        best_sim_threshold_id_rate_gteq_90 = similarity_thresholds[min_far_index_id_rate_gteq_90]

        return {
           'min_far_threshold': { "value": best_sim_threshold_far_lteq_1p, "index": max_iden_rate_index_for_far_lt_eq_1percent },
           'max_id_rate_threshold': { "value": best_sim_threshold_id_rate_gteq_90, "index": min_far_index_id_rate_gteq_90 }
        }
