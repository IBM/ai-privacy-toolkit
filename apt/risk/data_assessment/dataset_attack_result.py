from dataclasses import dataclass, field
from typing import Optional

import numpy as np

DEFAULT_DATASET_NAME = "dataset"


@dataclass
class DatasetAttackResult:
    pass


@dataclass
class DatasetAttackScore:
    dataset_name: str
    result: Optional[DatasetAttackResult] = None


@dataclass
class DatasetAttackResultMembership(DatasetAttackResult):
    member_probabilities: np.ndarray
    non_member_probabilities: np.ndarray
