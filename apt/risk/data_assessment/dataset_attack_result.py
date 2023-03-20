from dataclasses import dataclass
from typing import Optional

import numpy as np

DEFAULT_DATASET_NAME = "dataset"


@dataclass
class DatasetAttackResult:
    pass


@dataclass
class DatasetAttackScore:
    dataset_name: str
    risk_score: float
    result: Optional[DatasetAttackResult]


@dataclass
class DatasetAttackResultMembership(DatasetAttackResult):
    member_probabilities: np.ndarray
    non_member_probabilities: np.ndarray
