"""
This module implements privacy risk assessment of synthetic datasets based on the paper:
"GAN-Leaks: A Taxonomy of Membership Inference Attacks against Generative Models" by D. Chen, N. Yu, Y. Zhang, M. Fritz
published in Proceedings of the 2020 ACM SIGSAC Conference on Computer and Communications Security, 343–62, 2020.
https://doi.org/10.1145/3372297.3417238 and its implementation in https://github.com/DingfanChen/GAN-Leaks.
"""
from dataclasses import dataclass
from typing import Callable

import numpy as np
from sklearn.neighbors import NearestNeighbors

from apt.risk.data_assessment.attack_strategy_utils import KNNAttackStrategyUtils, DistributionValidationResult
from apt.risk.data_assessment.dataset_attack import DatasetAttackMembership, Config
from apt.risk.data_assessment.dataset_attack_result import DatasetAttackScore, DatasetAttackResultMembership, \
    DEFAULT_DATASET_NAME
from apt.utils.datasets import ArrayDataset


@dataclass
class DatasetAttackConfigMembershipKnnProbabilities(Config):
    """Configuration for DatasetAttackMembershipKnnProbabilities.

    Attributes:
        k: Number of nearest neighbors to search
        use_batches: Divide query samples into batches or not.
        batch_size:  Query sample batch size.
        compute_distance: A callable function, which takes two arrays representing 1D vectors as inputs and must return
            one value indicating the distance between those vectors.
            See 'metric' parameter in sklearn.neighbors.NearestNeighbors documentation.
        distance_params:  Additional keyword arguments for the distance computation function, see 'metric_params' in
            sklearn.neighbors.NearestNeighbors documentation.
        generate_plot: Generate or not an AUR ROC curve and persist it in a file
        distribution_comparison_alpha: the significance level of the statistical distribution test p-value.
                                       If p-value is less than alpha, then we reject the null hypothesis that the
                                       observed samples are drawn from the same distribution, and we claim that the
                                       distributions are different.
    """
    k: int = 5
    use_batches: bool = False
    batch_size: int = 10
    compute_distance: Callable = None
    distance_params: dict = None
    generate_plot: bool = False
    distribution_comparison_alpha: float = 0.05


@dataclass
class DatasetAttackScoreMembershipKnnProbabilities(DatasetAttackScore):
    """DatasetAttackMembershipKnnProbabilities privacy risk score.
    """
    roc_auc_score: float
    average_precision_score: float
    distributions_validation_result: DistributionValidationResult
    assessment_type: str = 'MembershipKnnProbabilities'  # to be used in reports

    def __init__(self, dataset_name: str, roc_auc_score: float, average_precision_score: float,
                 result: DatasetAttackResultMembership) -> None:
        """
        dataset_name:    dataset name to be used in reports
        roc_auc_score:   the area under the receiver operating characteristic curve (AUC ROC) to evaluate the attack
                          performance.
        average_precision_score: the proportion of predicted members that are correctly members
        result:          the result of the membership inference attack
        """
        super().__init__(dataset_name=dataset_name, risk_score=roc_auc_score, result=result)
        self.roc_auc_score = roc_auc_score
        self.average_precision_score = average_precision_score


class DatasetAttackMembershipKnnProbabilities(DatasetAttackMembership):
    """
         Privacy risk assessment for synthetic datasets based on Black-Box MIA attack using distances of
         members (training set) and non-members (holdout set) from their nearest neighbors in the synthetic dataset.
         By default, the Euclidean distance is used (L2 norm), but another ``compute_distance()`` method can be provided
         in configuration instead.
         The area under the receiver operating characteristic curve (AUC ROC) gives the privacy risk measure.
    """
    SHORT_NAME = 'MembershipKnnProbabilities'

    def __init__(self, original_data_members: ArrayDataset, original_data_non_members: ArrayDataset,
                 synthetic_data: ArrayDataset,
                 config: DatasetAttackConfigMembershipKnnProbabilities = DatasetAttackConfigMembershipKnnProbabilities(),
                 dataset_name: str = DEFAULT_DATASET_NAME,
                 categorical_features: list = None, **kwargs):
        """
        :param original_data_members: A container for the training original samples and labels. Should be encoded and
                                     scaled.
        :param original_data_non_members: A container for the holdout original samples and labels. Should be encoded and
                                          scaled.
        :param synthetic_data: A container for the synthetic samples and labels. Should be encoded and scaled.
        :param config: Configuration parameters to guide the attack, optional
        :param dataset_name: A name to identify this dataset, optional
        """
        attack_strategy_utils = KNNAttackStrategyUtils(config.use_batches, config.batch_size,
                                                       config.distribution_comparison_alpha, **kwargs)
        super().__init__(original_data_members, original_data_non_members, synthetic_data, config, dataset_name,
                         categorical_features, attack_strategy_utils)
        if config.compute_distance:
            self.knn_learner = NearestNeighbors(n_neighbors=config.k, algorithm='auto', metric=config.compute_distance,
                                                metric_params=config.distance_params)
        else:
            self.knn_learner = NearestNeighbors(n_neighbors=config.k, algorithm='auto')

    def short_name(self):
        return self.SHORT_NAME

    def assess_privacy(self) -> DatasetAttackScoreMembershipKnnProbabilities:
        """
        Membership Inference Attack which calculates probabilities of member and non-member samples to be generated by
        the synthetic data generator.
        The assumption is that since the generative model is trained to approximate the training data distribution
        then the probability of a sample to be a member of the training data should be proportional to the probability
        that the query sample can be generated by the generative model.
        So, if the probability that the query sample is generated by the generative model is large,
        it is more likely that the query sample was used to train the generative model. This probability is approximated
        by the Parzen window density estimation in ``probability_per_sample()``, computed from the NN distances from the
        query samples to the synthetic data samples.
        Before running the assessment, there is a validation that the distribution of the synthetic data is similar to
        that of the original data members and to that of the original data non-members.

        :return:
            Privacy score of the attack together with the attack result with the probabilities of member and
            non-member samples to be generated by the synthetic data generator based on the NN distances from the
            query samples to the synthetic data samples
            The result also contains the distribution validation result and a warning if the distributions are not
            similar.
        """
        distributions_validation_result = self.attack_strategy_utils.validate_distributions(
            self.original_data_members, self.original_data_non_members, self.synthetic_data, self.categorical_features)

        # nearest neighbor search
        self.attack_strategy_utils.fit(self.knn_learner, self.synthetic_data)

        # members query
        member_distances = self.attack_strategy_utils.find_knn(self.knn_learner, self.original_data_members)

        # non-members query
        non_member_distances = self.attack_strategy_utils.find_knn(self.knn_learner, self.original_data_non_members)

        member_proba = self.probability_per_sample(member_distances)
        non_member_proba = self.probability_per_sample(non_member_distances)
        result = DatasetAttackResultMembership(member_probabilities=member_proba,
                                               non_member_probabilities=non_member_proba)

        score = self.calculate_privacy_score(result, self.config.generate_plot)
        score.distributions_validation_result = distributions_validation_result
        return score

    def calculate_privacy_score(self, dataset_attack_result: DatasetAttackResultMembership,
                                generate_plot: bool = False) -> DatasetAttackScoreMembershipKnnProbabilities:
        """
        Evaluate privacy score from the probabilities of member and non-member samples to be generated by the synthetic
        data generator. The probabilities are computed by the ``assess_privacy()`` method.
        :param dataset_attack_result attack result containing probabilities of member and non-member samples to be
                generated by the synthetic data generator
        :param generate_plot generate AUC ROC curve plot and persist it
        :return:
            score of the attack, based on distance-based probabilities - mainly the ROC AUC score
        """
        member_proba, non_member_proba = \
            dataset_attack_result.member_probabilities, dataset_attack_result.non_member_probabilities
        fpr, tpr, threshold, auc, ap = self.calculate_metrics(member_proba, non_member_proba)
        score = DatasetAttackScoreMembershipKnnProbabilities(self.dataset_name,
                                                             result=dataset_attack_result,
                                                             roc_auc_score=auc, average_precision_score=ap)
        if generate_plot:
            self.plot_roc_curve(self.dataset_name, member_proba, non_member_proba)
        return score

    @staticmethod
    def probability_per_sample(distances: np.ndarray):
        """
        For every sample represented by its distance from the query sample to its KNN in synthetic data,
        computes the probability of the synthetic data to be part of the query dataset.
        :param distances: distance between every query sample in batch to its KNNs among synthetic samples, a numpy
        array of size (n, k) with n being the number of samples, k - the number of KNNs
        :return:
            probability estimates of the query samples being generated and so - of being part of the synthetic set, a
            numpy array of size (n,)
        """
        return np.average(np.exp(-distances), axis=1)
