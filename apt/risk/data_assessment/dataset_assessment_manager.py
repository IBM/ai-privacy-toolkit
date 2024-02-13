from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from apt.risk.data_assessment.dataset_attack_membership_knn_probabilities import \
    DatasetAttackConfigMembershipKnnProbabilities, DatasetAttackMembershipKnnProbabilities
from apt.risk.data_assessment.dataset_attack_result import DatasetAttackScore, DEFAULT_DATASET_NAME
from apt.risk.data_assessment.dataset_attack_whole_dataset_knn_distance import \
    DatasetAttackConfigWholeDatasetKnnDistance, DatasetAttackWholeDatasetKnnDistance
from apt.utils.datasets import ArrayDataset
from apt.risk.data_assessment.dataset_attack_membership_classification import \
    DatasetAttackConfigMembershipClassification, DatasetAttackMembershipClassification


@dataclass
class DatasetAssessmentManagerConfig:
    """
    Configuration for DatasetAssessmentManager.
    :param persist_reports: save assessment results to filesystem, or not.
    :param timestamp_reports: if persist_reports is True, then define if create a separate report for each timestamp,
                              or append to the same reports
    :param generate_plots: generate and visualize plots as part of assessment, or not..
    """
    persist_reports: bool = False
    timestamp_reports: bool = False
    generate_plots: bool = False


class DatasetAssessmentManager:
    """
    The main class for running dataset assessment attacks.
    """
    attack_scores = defaultdict(list)

    def __init__(self, config: Optional[DatasetAssessmentManagerConfig] = DatasetAssessmentManagerConfig) -> None:
        """
        :param config: Configuration parameters to guide the dataset assessment process
        """
        self.config = config

    def assess(self, original_data_members: ArrayDataset, original_data_non_members: ArrayDataset,
               synthetic_data: ArrayDataset, dataset_name: str = DEFAULT_DATASET_NAME, categorical_features: list = [])\
            -> list[DatasetAttackScore]:
        """
        Do dataset privacy risk assessment by running dataset attacks, and return their scores. All data is assumed
        to be encoded and scaled.

        :param original_data_members: A container for the training original samples and labels,
            only samples are used in the assessment
        :param original_data_non_members: A container for the holdout original samples and labels,
            only samples are used in the assessment
        :param synthetic_data: A container for the synthetic samples and labels, only samples are used in the assessment
        :param dataset_name: A name to identify this dataset, optional
        :param categorical_features: A list of categorical feature names or numbers

        :return:
            a list of dataset attack risk scores
        """
        # Create attacks
        config_gl = DatasetAttackConfigMembershipKnnProbabilities(use_batches=False,
                                                                  generate_plot=self.config.generate_plots)
        attack_gl = DatasetAttackMembershipKnnProbabilities(original_data_members,
                                                            original_data_non_members,
                                                            synthetic_data,
                                                            config_gl,
                                                            dataset_name, categorical_features)

        config_h = DatasetAttackConfigWholeDatasetKnnDistance(use_batches=False)
        attack_h = DatasetAttackWholeDatasetKnnDistance(original_data_members, original_data_non_members,
                                                        synthetic_data, config_h, dataset_name, categorical_features)

        config_mc = DatasetAttackConfigMembershipClassification(classifier_type='LogisticRegression',
                                                                # 'RandomForestClassifier',
                                                                threshold=0.9)
        attack_mc = DatasetAttackMembershipClassification(original_data_members, original_data_non_members,
                                                          synthetic_data, config_mc, dataset_name)

        attack_list = [
            (attack_gl, attack_gl.short_name()),  # "MembershipKnnProbabilities"
            (attack_h, attack_h.short_name()),    # "WholeDatasetKnnDistance"
            (attack_mc, attack_mc.short_name()),  # "MembershipClassification"
        ]

        for i, (attack, attack_name) in enumerate(attack_list):
            print(f"Running {attack_name} attack on {dataset_name}")
            score = attack.assess_privacy()
            self.attack_scores[attack_name].append(score)

        return self.attack_scores

    def dump_all_scores_to_files(self):
        """
         Save assessment results to filesystem.
         """
        if self.config.persist_reports:
            time_str = time.strftime("%Y%m%d-%H%M%S")
            for i, (attack_name, attack_scores) in enumerate(self.attack_scores.items()):
                if self.config.timestamp_reports:
                    results_log_file = f"{time_str}_{attack_name}_results.log.csv"
                else:
                    results_log_file = f"{attack_name}_results.log.csv"
                run_results_df = (pd.DataFrame(attack_scores).drop('result', axis=1, errors='ignore').
                                  drop('distributions_validation_result', axis=1, errors='ignore'))
                run_results_df.to_csv(results_log_file, header=True, encoding='utf-8', index=False, mode='w')
