from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

from apt.risk.data_assessment.dataset_attack_membership_knn_probabilities import \
    DatasetAttackConfigMembershipKnnProbabilities, DatasetAttackMembershipKnnProbabilities
from apt.risk.data_assessment.dataset_attack_result import DatasetAttackScore, DEFAULT_DATASET_NAME
from apt.risk.data_assessment.dataset_attack_whole_dataset_knn_distance import \
    DatasetAttackConfigWholeDatasetKnnDistance, DatasetAttackWholeDatasetKnnDistance
from apt.utils.datasets import ArrayDataset


@dataclass
class DatasetAssessmentManagerConfig:
    """
    Configuration for DatasetAssessmentManager.

    :param persist_reports: Whether to save assessment results to filesystem.
    :param generate_plots: Whether to generate and visualize plots as part of assessment.
    """
    persist_reports: bool = False
    generate_plots: bool = False


class DatasetAssessmentManager:
    """
    The main class for running dataset assessment attacks.

    :param config: Configuration parameters to guide the dataset assessment process
    """
    attack_scores_per_record_knn_probabilities: list[DatasetAttackScore] = []
    attack_scores_whole_dataset_knn_distance: list[DatasetAttackScore] = []

    def __init__(self, config: Optional[DatasetAssessmentManagerConfig] = DatasetAssessmentManagerConfig) -> None:
        self.config = config

    def assess(self, original_data_members: ArrayDataset, original_data_non_members: ArrayDataset,
               synthetic_data: ArrayDataset, dataset_name: str = DEFAULT_DATASET_NAME) -> list[DatasetAttackScore]:
        """
        Do dataset privacy risk assessment by running dataset attacks, and return their scores.

        :param original_data_members: A container for the training original samples and labels,
            only samples are used in the assessment
        :param original_data_non_members: A container for the holdout original samples and labels,
            only samples are used in the assessment
        :param synthetic_data: A container for the synthetic samples and labels, only samples are used in the assessment
        :param dataset_name: A name to identify this dataset, optional

        :return:
            a list of dataset attack risk scores
        """
        config_gl = DatasetAttackConfigMembershipKnnProbabilities(use_batches=False,
                                                                  generate_plot=self.config.generate_plots)
        attack_gl = DatasetAttackMembershipKnnProbabilities(original_data_members,
                                                            original_data_non_members,
                                                            synthetic_data,
                                                            config_gl,
                                                            dataset_name)

        score_gl = attack_gl.assess_privacy()
        self.attack_scores_per_record_knn_probabilities.append(score_gl)

        config_h = DatasetAttackConfigWholeDatasetKnnDistance(use_batches=False)
        attack_h = DatasetAttackWholeDatasetKnnDistance(original_data_members, original_data_non_members,
                                                        synthetic_data, config_h, dataset_name)

        score_h = attack_h.assess_privacy()
        self.attack_scores_whole_dataset_knn_distance.append(score_h)
        return [score_gl, score_h]

    def dump_all_scores_to_files(self):
        """
        Save assessment results to filesystem.
        """
        if self.config.persist_reports:
            results_log_file = "_results.log.csv"
            self._dump_scores_to_file(self.attack_scores_per_record_knn_probabilities,
                                     "per_record_knn_probabilities" + results_log_file, True)
            self._dump_scores_to_file(self.attack_scores_whole_dataset_knn_distance,
                                     "whole_dataset_knn_distance" + results_log_file, True)

    @staticmethod
    def _dump_scores_to_file(attack_scores: list[DatasetAttackScore], filename: str, header: bool):
        run_results_df = pd.DataFrame(attack_scores).drop('result', axis=1, errors='ignore')  # don't serialize result
        run_results_df.to_csv(filename, header=header, encoding='utf-8', index=False, mode='w')  # Overwrite
