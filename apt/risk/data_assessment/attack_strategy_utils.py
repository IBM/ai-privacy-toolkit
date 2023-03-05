import abc

import numpy as np
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from apt.utils.datasets import ArrayDataset


class AttackStrategyUtils(abc.ABC):
    """
        Abstract base class for common utilities of various privacy attack strategies.
    """
    ...


class KNNAttackStrategyUtils(AttackStrategyUtils):
    """
         Common utilities for attack strategy based on KNN distances.
    """

    def __init__(self, k: int, use_batches: bool = False, batch_size: int = 0) -> None:
        """
        :param k: How many nearest neighbors to search
        :param use_batches: Use batches with a progress meter or not when finding KNNs for query set
        :param batch_size: if use_batches=True, the size of batch_size should be > 0
        """
        self.k = k
        self.use_batches = use_batches
        self.batch_size = batch_size
        if use_batches:
            if batch_size < 1:
                raise ValueError(f"When using batching batch_size should be > 0, and not {batch_size}")

    def fit(self, dataset: ArrayDataset, knn_learner: NearestNeighbors):
        knn_learner.fit(dataset.get_samples())

    def find_knn(self, query_samples: ArrayDataset, knn_learner: NearestNeighbors, distance_processor=None):
        """
        Main nearest neighbor search function on synthetic data.
        :param query_samples: query samples
        :param knn_learner: unsupervised learner for implementing neighbor searches
        :param distance_processor: function for processing the distance into another more relevant metric per sample.
            Its input is an array representing distances (the distances returned by NearestNeighbors.kneighbors() ),
            and the output should be another array with distance-based values that enable to compute the final score
        :return:
            distances of the query samples to their nearest neighbors, or a metric based on that distance and calculated
            by the distance_processor function
        """
        samples = query_samples.get_samples()
        if not self.use_batches:
            distances, _ = knn_learner.kneighbors(samples, self.k, return_distance=True)
            if distance_processor:
                return distance_processor(distances)
            else:
                return distances

        probabilities = []
        for i in tqdm(range(len(samples) // self.batch_size)):
            x_batch = samples[i * self.batch_size:(i + 1) * self.batch_size]
            x_batch = np.reshape(x_batch, [self.batch_size, -1])

            # dist_batch: distance between every query sample in batch to its KNNs among training samples
            dist_batch, _ = knn_learner.kneighbors(x_batch, self.k, return_distance=True)

            # The probability of each sample to be generated
            if distance_processor:
                probability_per_sample_batch = distance_processor(dist_batch)
                probabilities.append(probability_per_sample_batch)
            else:
                probabilities.append(dist_batch)
        return np.concatenate(probabilities)
