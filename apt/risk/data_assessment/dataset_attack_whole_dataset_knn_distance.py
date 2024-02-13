"""
This module implements privacy risk assessment of synthetic datasets based on the papers
"Data Synthesis based on Generative Adversarial Networks." by N. Park, M. Mohammadi, K. Gorde, S. Jajodia, H. Park,
and Y. Kim in International Conference on Very Large Data Bases (VLDB), 2018.
and "Holdout-Based Fidelity and Privacy Assessment of Mixed-Type Synthetic Data" by M. Platzer and T. Reutterer.
and on a variation of its reference implementation in https://github.com/mostly-ai/paper-fidelity-accuracy.
"""
from dataclasses import dataclass

import numpy as np
from sklearn.neighbors import NearestNeighbors

from apt.risk.data_assessment.attack_strategy_utils import KNNAttackStrategyUtils, DistributionValidationResult
from apt.risk.data_assessment.dataset_attack import Config, DatasetAttack
from apt.risk.data_assessment.dataset_attack_result import DatasetAttackScore, DEFAULT_DATASET_NAME
from apt.utils.datasets import ArrayDataset

K = 1  # Number of nearest neighbors to search. For DCR we need only the nearest neighbor.


@dataclass
class DatasetAttackConfigWholeDatasetKnnDistance(Config):
    """Configuration for DatasetAttackWholeDatasetKnnDistance.

    Attributes:
        use_batches:  Divide query samples into batches or not.
        batch_size:   Query sample batch size.
        compute_distance: A callable function, which takes two arrays representing 1D vectors as inputs and must return
            one value indicating the distance between those vectors.
            See 'metric' parameter in sklearn.neighbors.NearestNeighbors documentation.
        distance_params:  Additional keyword arguments for the distance computation function, see 'metric_params' in
            sklearn.neighbors.NearestNeighbors documentation.
        distribution_comparison_alpha: the significance level of the statistical distribution test p-value.
                                       If p-value is less than alpha, then we reject the null hypothesis that the
                                       observed samples are drawn from the same distribution, and we claim that the
                                       distributions are different.
    """
    use_batches: bool = False
    batch_size: int = 10
    compute_distance: callable = None
    distance_params: dict = None
    distribution_comparison_alpha: float = 0.05
    distribution_comparison_numeric_test: str = 'KS',
    distribution_comparison_categorical_test: str = 'CHI'


@dataclass
class DatasetAttackScoreWholeDatasetKnnDistance(DatasetAttackScore):
    """DatasetAttackWholeDatasetKnnDistance privacy risk score.
    """
    share: float
    distributions_validation_result: DistributionValidationResult
    assessment_type: str = 'WholeDatasetKnnDistance'  # to be used in reports

    def __init__(self, dataset_name: str, share: float) -> None:
        """
        dataset_name:    dataset name to be used in reports
        share : the share of synthetic records closer to the training than the holdout dataset.
                A value of 0.5 or close to it means good privacy.
        """
        super().__init__(dataset_name=dataset_name, risk_score=share, result=None)
        self.share = share


class DatasetAttackWholeDatasetKnnDistance(DatasetAttack):
    """
         Privacy risk assessment for synthetic datasets based on distances of synthetic data records from
         members (training set) and non-members (holdout set). The privacy risk measure is the share of synthetic
         records closer to the training than the holdout dataset.
         By default, the Euclidean distance is used (L2 norm), but another compute_distance() method can be provided in
         configuration instead.
    """
    SHORT_NAME = 'WholeDatasetKnnDistance'

    def __init__(self, original_data_members: ArrayDataset, original_data_non_members: ArrayDataset,
                 synthetic_data: ArrayDataset,
                 config: DatasetAttackConfigWholeDatasetKnnDistance = DatasetAttackConfigWholeDatasetKnnDistance(),
                 dataset_name: str = DEFAULT_DATASET_NAME, categorical_features: list = None, **kwargs):
        """
        :param original_data_members: A container for the training original samples and labels. Should be encoded and
                                      scaled.
        :param original_data_non_members: A container for the holdout original samples and labels. Should be encoded
                                          and scaled.
        :param synthetic_data: A container for the synthetic samples and labels. Should be encoded and scaled.
        :param config: Configuration parameters to guide the assessment process, optional
        :param dataset_name: A name to identify this dataset, optional
        """
        attack_strategy_utils = KNNAttackStrategyUtils(config.use_batches, config.batch_size,
                                                       config.distribution_comparison_alpha, **kwargs)
        super().__init__(original_data_members, original_data_non_members, synthetic_data, config, dataset_name,
                         categorical_features, attack_strategy_utils)
        if config.compute_distance:
            self.knn_learner_members = NearestNeighbors(n_neighbors=K, metric=config.compute_distance,
                                                        metric_params=config.distance_params)
            self.knn_learner_non_members = NearestNeighbors(n_neighbors=K, metric=config.compute_distance,
                                                            metric_params=config.distance_params)
        else:
            self.knn_learner_members = NearestNeighbors(n_neighbors=K)
            self.knn_learner_non_members = NearestNeighbors(n_neighbors=K)

    def short_name(self):
        return self.SHORT_NAME

    def assess_privacy(self) -> DatasetAttackScoreWholeDatasetKnnDistance:
        """
        Calculate the share of synthetic records closer to the training than the holdout dataset, based on the
        DCR computed by 'calculate_distances()'.
        Before running the assessment, there is a validation that the distribution of the synthetic data is similar to
        that of the original data members and to that of the original data non-members.
        :return:
            score of the attack, based on the NN distances from the query samples to the synthetic data samples.
            The result also contains the distribution validation result and a warning if the distributions are not
            similar.
        """
        distributions_validation_result = self.attack_strategy_utils.validate_distributions(
            self.original_data_members, self.original_data_non_members, self.synthetic_data, self.categorical_features)

        member_distances, non_member_distances = self.calculate_distances()
        # distance of the synth. records to members and to non-members
        assert (len(member_distances) == len(non_member_distances))
        n_members = len(self.original_data_members.get_samples())
        n_non_members = len(self.original_data_non_members.get_samples())

        # percent of synth. records closer to members,
        # and distance ties are divided equally between members and non-members
        share = np.mean(member_distances < non_member_distances) + (n_members / (n_members + n_non_members)) * np.mean(
            member_distances == non_member_distances)
        score = DatasetAttackScoreWholeDatasetKnnDistance(self.dataset_name, share=share)
        score.distributions_validation_result = distributions_validation_result
        return score

    def calculate_distances(self):
        """
        Calculate member and non-member query probabilities, based on their distance to their KNN among
        synthetic samples. This distance is called distance to the closest record (DCR), as defined by
        N. Park et. al. in "Data Synthesis based on Generative Adversarial Networks."

        :return:
            member_distances - distances of each synthetic data member from its nearest training sample
            non_member_distances - distances of each synthetic data member from its nearest validation sample
        """
        # nearest neighbor search
        self.attack_strategy_utils.fit(self.knn_learner_members, self.original_data_members)
        self.attack_strategy_utils.fit(self.knn_learner_non_members, self.original_data_non_members)

        # distances of the synthetic data from the member and non-member samples
        member_distances = self.attack_strategy_utils.find_knn(self.knn_learner_members, self.synthetic_data)
        non_member_distances = self.attack_strategy_utils.find_knn(self.knn_learner_non_members, self.synthetic_data)

        return member_distances, non_member_distances
