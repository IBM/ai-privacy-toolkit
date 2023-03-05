"""
This module defines the interface for privacy risk assessment of synthetic datasets.
"""
import abc
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.metrics import RocCurveDisplay

from apt.risk.data_assessment.attack_strategy_utils import AttackStrategyUtils
from apt.risk.data_assessment.dataset_attack_result import DatasetAttackScore, DatasetAttackResultPerRecord, \
    DatasetAttackResult
from apt.utils.datasets import ArrayDataset


class Config:
    """
        The base class for dataset attack configurations
    """
    ...


class DatasetAttack(abc.ABC):
    """
         The interface for performing privacy risk assessment for synthetic datasets.
    """

    def __init__(self, original_data_members: ArrayDataset, original_data_non_members: ArrayDataset,
                 synthetic_data: ArrayDataset, dataset_name: str, attack_strategy_utils: AttackStrategyUtils,
                 config: Optional[Config] = Config()) -> None:
        """
        :param original_data_members: A container for the training original samples and labels
        :param original_data_non_members: A container for the holdout original samples and labels
        :param synthetic_data: A container for the synthetic samples and labels
        :param dataset_name: A name to identify the dataset under attack
        :param attack_strategy_utils: Utils for use with the attack strategy
        :param config: Configuration parameters to guide the assessment process such as which attack
               frameworks to use, optional
        """

        self.original_data_members = original_data_members
        self.original_data_non_members = original_data_non_members
        self.synthetic_data = synthetic_data
        self.dataset_name = dataset_name
        self.attack_strategy_utils = attack_strategy_utils
        self.config = config

    @abc.abstractmethod
    def assess_privacy(self) -> Union[DatasetAttackScore, DatasetAttackResult]:
        """
        Assess the privacy of the dataset
        :return:
            result: Union[DatasetAttackScore, DatasetAssessmentResult] can be either the final privacy attack score,
            or an intermediate attack result, which can be translated into a privacy score if needed
        """
        ...


class DatasetAttackPerRecord(DatasetAttack):
    """
         An abstract base class for performing privacy risk assessment for synthetic datasets on a per-record level.
    """

    @abc.abstractmethod
    def assess_privacy(self) -> DatasetAttackResultPerRecord:
        """
        Assess the privacy of the dataset
        :return:
            result: DatasetAssessmentResult
        """
        ...

    @abc.abstractmethod
    def calculate_privacy_score(self, dataset_attack_result: DatasetAttackResultPerRecord,
                                generate_plot=False) -> DatasetAttackScore:
        """
        Calculate dataset privacy score based on the result of the privacy assessment
        :return:
            result: DatasetAttackScore
        """
        ...

    def plot_roc_curve(self, pos_probabilities, neg_probabilities, name_prefix=""):
        """
        Plot ROC curve
        :param pos_probabilities: loss of the positive samples, the training data
        :param neg_probabilities: loss of the negative samples, the hold-out data
        :param name_prefix: name prefix for the ROC curve plot
        """
        labels = np.concatenate((np.zeros((len(neg_probabilities),)), np.ones((len(pos_probabilities),))))
        results = np.concatenate((neg_probabilities, pos_probabilities))
        svc_disp = RocCurveDisplay.from_predictions(labels, results)
        svc_disp.plot()
        plt.plot([0, 1], [0, 1], color="navy", linewidth=2, linestyle="--", label='No skills')
        plt.title('ROC curve')
        plt.savefig(f'{name_prefix}{self.dataset_name}_roc_curve.png')

    @staticmethod
    def calculate_roc_score(pos_probabilities, neg_probabilities):
        """
        Plot ROC curve
        :param pos_probabilities: probability estimates of the positive samples, the training data
        :param neg_probabilities: probability estimates of the negative samples, the hold-out data
        :return:
            fpr: False Positive rate
            tpr: True Positive rate
            threshold: threshold
            auc: area under the Receiver Operating Characteristic Curve
            ap: average precision score
        """
        labels = np.concatenate((np.zeros((len(neg_probabilities),)), np.ones((len(pos_probabilities),))))
        results = np.concatenate((neg_probabilities, pos_probabilities))
        fpr, tpr, threshold = metrics.roc_curve(labels, results, pos_label=1)
        auc = metrics.roc_auc_score(labels, results)
        ap = metrics.average_precision_score(labels, results)
        return fpr, tpr, threshold, auc, ap


class DatasetAttackWhole(DatasetAttack):
    """
         An abstract base class for performing privacy risk assessment for synthetic datasets on a whole-dataset level.
    """

    @abc.abstractmethod
    def assess_privacy(self) -> DatasetAttackScore:
        """
        Assess the privacy of the dataset
        :return:
            result: DatasetAssessmentResult
        """
        ...
