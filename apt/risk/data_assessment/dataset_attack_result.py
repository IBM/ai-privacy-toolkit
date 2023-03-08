from dataclasses import dataclass, field

import numpy as np


@dataclass
class DatasetAttackScore:
    dataset_name: str


@dataclass
class DatasetAttackResult:
    dataset_name: str


@dataclass
class DatasetAttackScoreWithResult(DatasetAttackScore):
    result: DatasetAttackResult = field(repr=False)


@dataclass
class DatasetAttackResultPerRecord(DatasetAttackResult):
    positive_probabilities: np.ndarray
    negative_probabilities: np.ndarray
