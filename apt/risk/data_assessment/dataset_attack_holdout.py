"""
This module implements privacy risk assessment of synthetic datasets based on the paper
"Holdout-Based Fidelity and Privacy Assessment of Mixed-Type Synthetic Data" by M. Platzer and T. Reutterer.
and on a variation of its reference implementation in https://github.com/mostly-ai/paper-fidelity-accuracy.
"""
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.neighbors import NearestNeighbors

from apt.risk.data_assessment.attack_strategy_utils import KNNAttackStrategyUtils
from apt.risk.data_assessment.dataset_attack import DatasetAttackWhole, Config
from apt.risk.data_assessment.dataset_attack_result import DatasetAttackScore
from apt.utils.datasets import ArrayDataset

logger = logging.getLogger(__name__)


@dataclass
class DatasetAttackHoldoutConfig(Config):
    """Configuration for DatasetAttackHoldout.

    Attributes:
        k: Number of nearest neighbors to search
        use_batches:  Divide query samples into batches or not.
        batch_size:   Query sample batch size.
        compute_distance: A callable function, which takes two arrays representing 1D vectors as inputs and must return
            one value indicating the distance between those vectors.
        batch_size:   Additional keyword arguments for the distance computation function.
    """
    k: int = 1
    use_batches: bool = False
    batch_size: int = 10
    compute_distance: callable = None
    distance_params: dict = None


@dataclass
class DatasetAttackScoreHoldout(DatasetAttackScore):
    """Configuration for DatasetAttackHoldout.
    Attributes
    ----------
    share : the share of synthetic records closer to the training than the holdout dataset
    assessment_type : assessment type is 'Holdout', to be used in reports
    """
    share: float
    assessment_type: str = 'Holdout'


class DatasetAttackHoldout(DatasetAttackWhole):
    """
         Privacy risk assessment for synthetic datasets based on distances of synthetic data records from
         members (training set) and non-members (holdout set). The privacy risk measure is the share of synthetic
         records closer to the training than the holdout dataset.
    """

    def __init__(self, original_data_members: ArrayDataset, original_data_non_members: ArrayDataset,
                 synthetic_data: ArrayDataset, dataset_name: str,
                 config: Optional[DatasetAttackHoldoutConfig] = DatasetAttackHoldoutConfig()):
        """
        :param original_data_members: A container for the training original samples and labels
        :param original_data_non_members: A container for the holdout original samples and labels
        :param synthetic_data: A container for the synthetic samples and labels
        :param dataset_name: A name to identify this dataset
        :param config: Configuration parameters to guide the assessment process such as which attack
               frameworks to use, optional
        """
        attack_strategy_utils = KNNAttackStrategyUtils(config.k, config.use_batches, config.batch_size)
        super().__init__(original_data_members, original_data_non_members, synthetic_data, dataset_name,
                         attack_strategy_utils, config)
        if config.compute_distance:
            self.nn_obj_members = NearestNeighbors(n_neighbors=config.k, algorithm='auto',
                                                   metric=config.compute_distance,
                                                   metric_params=config.distance_params)
            self.nn_obj_non_members = NearestNeighbors(n_neighbors=config.k, algorithm='auto',
                                                       metric=config.compute_distance,
                                                       metric_params=config.distance_params)
        else:
            self.nn_obj_members = NearestNeighbors(n_neighbors=config.k, algorithm='auto')
            self.nn_obj_non_members = NearestNeighbors(n_neighbors=config.k, algorithm='auto')

    def assess_privacy(self) -> DatasetAttackScoreHoldout:
        """
        Calculate the share of synthetic records closer to the training than the holdout dataset
        :return:
            :result of the attack, based on the NN distances from the query samples to the synthetic data samples
        """
        member_distances, non_member_distances = self.calculate_distances()
        n_members = len(member_distances)
        n_non_members = len(non_member_distances)
        assert (n_members == n_non_members)
        share = np.mean(member_distances < non_member_distances) + (n_members / (n_members + n_non_members)) * np.mean(
            member_distances == non_member_distances)
        score = DatasetAttackScoreHoldout(self.dataset_name, share=share)
        return score

    def calculate_distances(self):
        """
        Calculate positive and negative query probabilities, based on their distance to their KNNs among
        synthetic samples.
        :return:
            pos_distances: distances of each synthetic data member from its nearest training samples
            neg_distances: distances of each synthetic data member from its nearest validation samples
        """
        # nearest neighbor search
        self.attack_strategy_utils.fit(self.original_data_members, self.nn_obj_members)
        self.attack_strategy_utils.fit(self.original_data_non_members, self.nn_obj_non_members)

        # distances of the synthetic data from the positive and negative samples (members and non-members)
        pos_distances = self.attack_strategy_utils.find_knn(self.synthetic_data, self.nn_obj_members)
        neg_distances = self.attack_strategy_utils.find_knn(self.synthetic_data, self.nn_obj_non_members)

        return pos_distances, neg_distances
