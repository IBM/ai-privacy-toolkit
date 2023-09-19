import abc
from dataclasses import dataclass

import numpy as np
from scipy import stats
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from pandas.api.types import is_numeric_dtype, is_categorical_dtype

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
        distributions_valid (bool): False if there are columns whose distribution is different between the datasets
        member_column_distribution_diff (list): Columns whose distribution is different between the member and the
                                                synthetic datasets
        non_member_column_distribution_diff (list): Columns whose distribution is different between the non-member and
                                                the synthetic datasets
    """
    distributions_valid: bool
    member_column_distribution_diff: list
    non_member_column_distribution_diff: list


class KNNAttackStrategyUtils(AttackStrategyUtils):
    """
         Common utilities for attack strategy based on KNN distances.
    """

    def __init__(self, use_batches: bool = False, batch_size: int = 10) -> None:
        """
        :param use_batches: Use batches with a progress meter or not when finding KNNs for query set
        :param batch_size: if use_batches=True, the size of batch_size should be > 0
        """
        self.use_batches = use_batches
        self.batch_size = batch_size
        if use_batches:
            if batch_size < 1:
                raise ValueError(f"When using batching batch_size should be > 0, and not {batch_size}")

    def fit(self, knn_learner: NearestNeighbors, dataset: ArrayDataset):
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
    def _column_statistical_test(df1_column_samples, df2_column_samples, column, is_categorical, is_numeric, test_type,
                                 alpha, differing_columns):
        if is_categorical(column):
            try:
                result = stats.chisquare(f_obs=df1_column_samples, f_exp=df1_column_samples)
            except ValueError as e:
                if str(e).startswith('For each axis slice, the sum of'):
                    print('Column', column, e)
                else:
                    raise
        elif is_numeric:
            if test_type == 'KS':
                result = stats.ks_2samp(df1_column_samples, df2_column_samples)
            elif test_type == 'CVM':
                result = stats.cramervonmises_2samp(df1_column_samples, df1_column_samples)
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

    @staticmethod
    def _columns_different_distributions(df1: ArrayDataset, df2: ArrayDataset,
                                         categorical_features: list = [],
                                         alpha=0.05, test_type='KS') -> list:
        differing_columns = []
        df1_samples = df1.get_samples()
        df2_samples = df2.get_samples()
        if df1.is_pandas:
            def is_categorical(col_name):
                col_name in categorical_features or is_categorical_dtype(df1_samples.dtypes[col_name])

            def is_numeric(col_name): is_numeric_dtype(df1_samples.dtypes[col_name])
            for name, _ in df1_samples.items():
                KNNAttackStrategyUtils._column_statistical_test(df1_samples[name], df2_samples[name], name,
                                                                is_categorical, is_numeric(df1_samples.dtypes[name]),
                                                                test_type, alpha, differing_columns)
        else:
            is_df1_numeric_dtype = np.issubdtype(df1_samples.dtype, int) or np.issubdtype(df1_samples.dtype, float)
            def is_categorical(col_name): col_name in categorical_features
            for i, column in enumerate(df1_samples.T):
                KNNAttackStrategyUtils._column_statistical_test(df1_samples[:, i], df2_samples[:, i], i,
                                                                is_categorical, is_df1_numeric_dtype, test_type, alpha,
                                                                differing_columns)
        return differing_columns

    def validate_distributions(self, original_data_members: ArrayDataset, original_data_non_members: ArrayDataset,
                               synthetic_data: ArrayDataset, categorical_features: list = None):
        """
        Validate column distributions are similar between the datasets.
        :param original_data_members: A container for the training original samples and labels
        :param original_data_non_members: A container for the holdout original samples and labels
        :param synthetic_data: A container for the synthetic samples and labels
        :param categorical_features: a list of categorical features of the datasets
        :return:
            DistributionValidationResult
        """
        member_column_distribution_diff = self._columns_different_distributions(synthetic_data,
                                                                                original_data_members,
                                                                                categorical_features)
        non_member_column_distribution_diff = self._columns_different_distributions(synthetic_data,
                                                                                    original_data_non_members,
                                                                                    categorical_features)
        if not member_column_distribution_diff and not non_member_column_distribution_diff:
            return DistributionValidationResult(distributions_valid=True,
                                                member_column_distribution_diff=[],
                                                non_member_column_distribution_diff=[])

        return DistributionValidationResult(distributions_valid=False,
                                            member_column_distribution_diff=member_column_distribution_diff,
                                            non_member_column_distribution_diff=non_member_column_distribution_diff)
