from dataclasses import dataclass
from typing import Optional

import pandas as pd

from apt.risk.data_assessment.dataset_attack_gan_leaks import DatasetAttackGanLeaksConfig, DatasetAttackGanLeaks, \
    DatasetAttackScoreGanLeaks
from apt.risk.data_assessment.dataset_attack_holdout import DatasetAttackHoldoutConfig, DatasetAttackHoldout, \
    DatasetAttackScoreHoldout
from apt.utils.datasets import ArrayDataset


@dataclass
class DatasetAssessmentManagerConfig:
    persist_reports: bool = False
    generate_plots: bool = False


class DatasetAssessmentManager:
    """
    The main class for running dataset assessment attacks.
    """
    gan_leaks_attack_scores = []
    holdout_attack_scores = []

    def __init__(self, config: Optional[DatasetAssessmentManagerConfig] = DatasetAssessmentManagerConfig) -> None:
        """
        :param config: Configuration parameters to guide the dataset assessment process
        """
        self.config = config

    def assess(self, original_data_members: ArrayDataset, original_data_non_members: ArrayDataset,
               synthetic_data: ArrayDataset, dataset_name: str = "dataset") -> (
            DatasetAttackScoreGanLeaks, DatasetAttackScoreHoldout):
        config_gl = DatasetAttackGanLeaksConfig(use_batches=False, k=5)
        mgr = DatasetAttackGanLeaks(original_data_members,
                                    original_data_non_members,
                                    synthetic_data,
                                    dataset_name,
                                    config_gl)

        result = mgr.assess_privacy()
        score_g = mgr.calculate_privacy_score(result, generate_plot=self.config.generate_plots)
        self.gan_leaks_attack_scores.append(score_g)

        config_h = DatasetAttackHoldoutConfig(use_batches=False, k=5)
        mgr_h = DatasetAttackHoldout(original_data_members, original_data_non_members, synthetic_data,
                                     dataset_name,
                                     config_h)

        score_h = mgr_h.assess_privacy()
        self.holdout_attack_scores.append(score_h)
        return score_g, score_h

    def dump_all_scores_to_files(self):
        if self.config.persist_reports:
            results_log_file = "_results.log.csv"
            self.dump_scores_to_file(self.gan_leaks_attack_scores, "gan_leaks" + results_log_file, True)
            self.dump_scores_to_file(self.holdout_attack_scores, "holdout" + results_log_file, True)

    @staticmethod
    def dump_scores_to_file(attack_scores, filename, header: bool):
        run_results_df = pd.DataFrame(attack_scores)
        run_results_df.to_csv(filename, header=header, encoding='utf-8', index=False, mode='w')  # Overwrite
