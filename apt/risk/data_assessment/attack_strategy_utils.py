import abc
from dataclasses import dataclass

import numpy as np
from scipy import stats
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from apt.utils.datasets import ArrayDataset


class AttackStrategyUtils(abc.ABC):
    """
        Abstract base class for common utilities of various privacy attack strategies.
    """
    pass


@dataclass
class DistributionValidationResult:
    """Holds the result of the validation of distributions similarities.

    Attributes:
        distributions_validated : False if distribution validation failed for some reason, and no conclusion was drawn
        distributions_valid: False if there are columns whose distribution is different between the datasets
        member_column_distribution_diff (list): Columns whose distribution is different between the member and the
                                                synthetic datasets
        non_member_column_distribution_diff (list): Columns whose distribution is different between the non-member and
                                                the synthetic datasets
    """
    distributions_validated: bool
    distributions_valid: bool
    member_column_distribution_diff: list
    non_member_column_distribution_diff: list


class KNNAttackStrategyUtils(AttackStrategyUtils):
    """
         Common utilities for attack strategy based on KNN distances.
    """

    def __init__(self, use_batches: bool = False, batch_size: int = 10, distribution_comparison_alpha: float = 0.05,
                 distribution_comparison_numeric_test: str = 'KS',
                 distribution_comparison_categorical_test: str = 'CHI') -> None:
        """
        :param use_batches: Use batches with a progress meter or not when finding KNNs for query set
        :param batch_size: if use_batches is True, the size of batch_size should be > 0
        :param distribution_comparison_alpha: the significance level of the statistical distribution test pvalue.
                                              If p-value is less than alpha, then we reject the null hypothesis that the
                                              observed samples are drawn from the same distribution and we claim that
                                              the distributions are different.
        :param distribution_comparison_numeric_test: Type of test to compare distributions of numeric columns. Can be:
                                                    'KS' for the two-sample Kolmogorov-Smirnov test for goodness of fit,
                                                    'CVM' for the two-sample Cramér-von Mises test for goodness of fit,
                                                    'AD' for the Anderson-Darling test for 2-samples,
                                                    'ES' for the Epps-Singleton (ES) test statistic. The default is 'KS'
        :param distribution_comparison_categorical_test: Type of test to compare distributions of categorical columns.
                                                        Can be:
                                                        'CHI' for the one-way chi-square test,
                                                        'AD' for The Anderson-Darling test for 2-samples,
                                                        'ES' for the Epps-Singleton (ES) test statistic.
                                                        The default is 'ES'.
        """
        self.use_batches = use_batches
        self.batch_size = batch_size
        if use_batches:
            if batch_size < 1:
                raise ValueError(f"When using batching batch_size should be > 0, and not {batch_size}")
        self.distribution_comparison_alpha = distribution_comparison_alpha
        self.distribution_comparison_numeric_test = distribution_comparison_numeric_test
        self.distribution_comparison_categorical_test = distribution_comparison_categorical_test

    def fit(self, knn_learner: NearestNeighbors, dataset: ArrayDataset):
        """
        Fit the KNN learner.

        :param knn_learner: The KNN model to fit.
        :param dataset: The training set to fit the model on.
        """
        knn_learner.fit(dataset.get_samples())

    def find_knn(self, knn_learner: NearestNeighbors, query_samples: ArrayDataset, distance_processor=None):
        """
        Nearest neighbor search function.

        :param query_samples: query samples, to which nearest neighbors are to be found
        :param knn_learner: unsupervised learner for implementing neighbor searches, after it was fitted
        :param distance_processor: function for processing the distance into another more relevant metric per sample.
            Its input is an array representing distances (the distances returned by NearestNeighbors.kneighbors() ), and
            the output should be another array with distance-based values that enable to compute the final risk score
        :return:
            distances of the query samples to their nearest neighbors, or a metric based on that distance and calculated
            by the distance_processor function
        """
        samples = query_samples.get_samples()
        if not self.use_batches:
            distances, _ = knn_learner.kneighbors(samples, return_distance=True)
            if distance_processor:
                return distance_processor(distances)
            else:
                return distances

        distances = []
        for i in tqdm(range(len(samples) // self.batch_size)):
            x_batch = samples[i * self.batch_size:(i + 1) * self.batch_size]
            x_batch = np.reshape(x_batch, [self.batch_size, -1])

            # dist_batch: distance between every query sample in batch to its KNNs among training samples
            dist_batch, _ = knn_learner.kneighbors(x_batch, return_distance=True)

            # The probability of each sample to be generated
            if distance_processor:
                distance_based_metric_per_sample_batch = distance_processor(dist_batch)
                distances.append(distance_based_metric_per_sample_batch)
            else:
                distances.append(dist_batch)
        return np.concatenate(distances)

    @staticmethod
    def _column_statistical_test(df1_column_samples, df2_column_samples, column, is_categorical, is_numeric,
                                 numeric_test_type, categorical_test_type, alpha, differing_columns):
        if is_categorical:
            test_type = categorical_test_type
            if test_type == 'CHI':
                try:
                    result = stats.chisquare(f_obs=df1_column_samples, f_exp=df1_column_samples)
                except ValueError as e:
                    if str(e).startswith('For each axis slice, the sum of'):
                        print('Column', column, ' the observed and expected sums are not the same,'
                                                'so cannot run distribution comparison test')
                        raise e
                    else:
                        raise
            elif test_type == 'AD':
                result = stats.anderson_ksamp([df1_column_samples, df2_column_samples], midrank=True)
            elif test_type == 'ES':
                result = stats.epps_singleton_2samp(df1_column_samples, df2_column_samples)
            else:
                raise ValueError('Unknown test type', test_type)
        elif is_numeric:
            test_type = numeric_test_type
            if test_type == 'KS':
                result = stats.ks_2samp(df1_column_samples, df2_column_samples)
            elif test_type == 'CVM':
                result = stats.cramervonmises_2samp(df1_column_samples, df1_column_samples)
            elif test_type == 'AD':
                result = stats.anderson_ksamp([df1_column_samples, df2_column_samples], midrank=True)
            elif test_type == 'ES':
                result = stats.epps_singleton_2samp(df1_column_samples, df2_column_samples)
            else:
                raise ValueError('Unknown test type', test_type)
        else:
            print(f'Skipping non-numeric and non-categorical column {column}')
            return
        print(
            f"{column}: {test_type} = {result.statistic:.4f} "
            f"(p-value = {result.pvalue:.3e}, are equal = {result.pvalue > 0.05})")
        if result.pvalue < alpha:
            # Reject H0, different distributions
            print(f"Distributions differ in column {column}, p-value: {result.pvalue}")
            differing_columns.append(column)
        else:
            # Accept H0, similar distributions
            print(f'Accept H0, similar distributions in column {column}')

    def _columns_different_distributions(self, df1: ArrayDataset, df2: ArrayDataset,
                                         categorical_features: list = []) -> list:
        differing_columns = []
        df1_samples = df1.get_samples()
        df2_samples = df2.get_samples()
        is_numeric = np.issubdtype(df1_samples.dtype, int) or np.issubdtype(df1_samples.dtype, float)

        for i, column in enumerate(df1_samples.T):
            is_categorical = i in categorical_features
            KNNAttackStrategyUtils._column_statistical_test(df1_samples[:, i], df2_samples[:, i], i,
                                                            is_categorical, is_numeric,
                                                            self.distribution_comparison_numeric_test,
                                                            self.distribution_comparison_categorical_test,
                                                            self.distribution_comparison_alpha, differing_columns)
        return differing_columns

    def validate_distributions(self, original_data_members: ArrayDataset, original_data_non_members: ArrayDataset,
                               synthetic_data: ArrayDataset, categorical_features: list = None):
        """
        Validate column distributions are similar between the datasets.
        One advantage of the ES test compared to the KS test is that is does not assume a continuous distribution.
        In [1], the authors conclude that the test also has a higher power than the KS test in many examples. They
        recommend the use of the ES test for discrete samples as well as continuous samples with at least 25
        observations each, whereas AD is recommended for smaller sample sizes in the continuous case.

        :param original_data_members: A container for the training original samples and labels
        :param original_data_non_members: A container for the holdout original samples and labels
        :param synthetic_data: A container for the synthetic samples and labels
        :param categorical_features: a list of categorical features of the datasets
        :return:
            DistributionValidationResult
        """
        try:
            member_column_distribution_diff = self._columns_different_distributions(synthetic_data,
                                                                                    original_data_members,
                                                                                    categorical_features)
            non_member_column_distribution_diff = self._columns_different_distributions(synthetic_data,
                                                                                        original_data_non_members,
                                                                                        categorical_features)
        except (ValueError, np.linalg.LinAlgError) as e:
            print("Failed to validate distributions", e)
            return DistributionValidationResult(distributions_validated=True,
                                                distributions_valid=False,
                                                member_column_distribution_diff=[],
                                                non_member_column_distribution_diff=[])

        if not member_column_distribution_diff and not non_member_column_distribution_diff:
            return DistributionValidationResult(distributions_validated=True,
                                                distributions_valid=True,
                                                member_column_distribution_diff=[],
                                                non_member_column_distribution_diff=[])

        return DistributionValidationResult(distributions_validated=True,
                                            distributions_valid=False,
                                            member_column_distribution_diff=member_column_distribution_diff,
                                            non_member_column_distribution_diff=non_member_column_distribution_diff)
