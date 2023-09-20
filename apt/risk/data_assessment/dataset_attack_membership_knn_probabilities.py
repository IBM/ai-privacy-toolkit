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
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

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
    """
    k: int = 5
    use_batches: bool = False
    batch_size: int = 10
    compute_distance: Callable = None
    distance_params: dict = None
    generate_plot: bool = False


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
                 categorical_features: list = None,
                 add_reference: bool = False, reference_synthetic_data: ArrayDataset = None):
        """
        :param original_data_members: A container for the training original samples and labels
        :param original_data_non_members: A container for the holdout original samples and labels
        :param synthetic_data: A container for the synthetic samples and labels
        :param config: Configuration parameters to guide the attack, optional
        :param dataset_name: A name to identify this dataset, optional
        """
        attack_strategy_utils = KNNAttackStrategyUtils(config.use_batches, config.batch_size)
        super().__init__(original_data_members, original_data_non_members, synthetic_data, config, dataset_name,
                         categorical_features, attack_strategy_utils)
        if config.compute_distance:
            self.knn_learner = NearestNeighbors(n_neighbors=config.k, algorithm='auto', metric=config.compute_distance,
                                                metric_params=config.distance_params)
        else:
            self.knn_learner = NearestNeighbors(n_neighbors=config.k, algorithm='auto')

        self.has_reference = add_reference
        if not add_reference:
            return

        if reference_synthetic_data:
            self.synthetic_data_ref = reference_synthetic_data
        else:
            # Y not used, but needed for ArrayDataset
            X_non_members, X_reference = \
                train_test_split(original_data_non_members.get_samples(), test_size=0.5, random_state=7)

            # ref_filename = "ref_data.csv"
            # test_filename = "test_data.csv"
            # if os.path.exists(ref_filename) and os.path.exists(test_filename):
            #     x_synth_ref = np.genfromtxt(ref_filename, delimiter=",")
            #     X_non_members = np.genfromtxt(test_filename, delimiter=",")
            # else:
            x_synth_ref = self.generate_synth_data(len(X_reference), n_components=10, original_data=X_reference)
            # np.savetxt(ref_filename, x_synth_ref, delimiter=",")
            # np.savetxt(test_filename, X_non_members, delimiter=",")

            self.original_data_non_members = ArrayDataset(X_non_members)
            self.synthetic_data_ref = ArrayDataset(x_synth_ref)
        if config.compute_distance:
            self.knn_learner_ref = NearestNeighbors(n_neighbors=config.k, algorithm='auto',
                                                    metric=config.compute_distance,
                                                    metric_params=config.distance_params)
        else:
            self.knn_learner_ref = NearestNeighbors(n_neighbors=config.k, algorithm='auto')

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

        :return:
            Privacy score of the attack together with the attack result with the probabilities of member and
            non-member samples to be generated by the synthetic data generator based on the NN distances from the
            query samples to the synthetic data samples
        """
        distributions_validation_result = self.attack_strategy_utils.validate_distributions(
            self.original_data_members, self.original_data_non_members, self.synthetic_data, self.categorical_features)

        # nearest neighbor search
        self.attack_strategy_utils.fit(self.knn_learner, self.synthetic_data)

        # members query
        member_distances = self.attack_strategy_utils.find_knn(self.knn_learner, self.original_data_members)

        # non-members query
        non_member_distances = self.attack_strategy_utils.find_knn(self.knn_learner, self.original_data_non_members)

        if self.has_reference:
            self.attack_strategy_utils.fit(self.knn_learner_ref, self.synthetic_data_ref)

            # members query
            member_distances_ref = self.attack_strategy_utils.find_knn(self.knn_learner_ref,
                                                                       self.original_data_members)

            # non-members query
            non_member_distances_ref = self.attack_strategy_utils.find_knn(self.knn_learner_ref,
                                                                           self.original_data_non_members)

            assert (len(member_distances) == len(member_distances_ref))
            assert (len(non_member_distances) == len(non_member_distances_ref))
            num_pos_samples = len(member_distances)
            num_neg_samples = len(non_member_distances)

            member_proba_calibrate = self.probability_per_sample(member_distances[:num_pos_samples] -
                                                                 member_distances_ref[:num_pos_samples])
            non_member_proba_calibrate = self.probability_per_sample(non_member_distances[:num_neg_samples] -
                                                                     non_member_distances_ref[:num_neg_samples])

            result = DatasetAttackResultMembership(member_probabilities=member_proba_calibrate,
                                                   non_member_probabilities=non_member_proba_calibrate)
        else:
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

    @staticmethod
    def generate_synth_data(n_samples, n_components, original_data):
        """
        Simple KDE synthetic data genrator: estimates the kernel density of data using a Gaussian kernel and then generates
        samples from this distribution
        """
        digit_data = original_data
        pca = PCA(n_components=n_components, whiten=False)
        data = pca.fit_transform(digit_data)
        params = {'bandwidth': np.logspace(-1, 1, 20)}
        grid = GridSearchCV(KernelDensity(), params, cv=5)
        grid.fit(data)

        kde_estimator = grid.best_estimator_

        new_data = kde_estimator.sample(n_samples, random_state=0)
        new_data = pca.inverse_transform(new_data)
        return new_data
