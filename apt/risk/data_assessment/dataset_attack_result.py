from dataclasses import dataclass, field

import numpy as np

DEFAULT_DATASET_NAME = "dataset"


@dataclass
class DatasetAttackScore:
    dataset_name: str


@dataclass
class DatasetAttackResult:
    pass


@dataclass(repr=False)
class DatasetAttackScoreWithResult(DatasetAttackScore):
    result: DatasetAttackResult = field(repr=False)


@dataclass
class DatasetAttackResultMembership(DatasetAttackResult):
    member_probabilities: np.ndarray
    non_member_probabilities: np.ndarray
