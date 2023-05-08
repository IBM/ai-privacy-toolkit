from dataclasses import dataclass
from typing import Optional

import numpy as np

DEFAULT_DATASET_NAME = "dataset"


@dataclass
class DatasetAttackResult:
    """
    Basic class for storing privacy risk assessment results.
    """
    pass


@dataclass
class DatasetAttackScore:
    """
    Basic class for storing privacy risk assessment scores.

    :param dataset_name: The name of the dataset that was assessed.
    :param risk_score: The privacy risk score.
    :param result: An optional list of more detailed results.
    """
    dataset_name: str
    risk_score: float
    result: Optional[DatasetAttackResult]


@dataclass
class DatasetAttackResultMembership(DatasetAttackResult):
    """
    Class for storing membership attack results.

    :param member_probabilities: The attack probabilities for member samples.
    :param non_member_probabilities: The attack probabilities for non-member samples.
    """
    member_probabilities: np.ndarray
    non_member_probabilities: np.ndarray
