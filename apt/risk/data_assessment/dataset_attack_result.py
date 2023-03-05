from dataclasses import dataclass

import numpy as np


@dataclass
class DatasetAttackResult:
    dataset_name: str


@dataclass
class DatasetAttackResultPerRecord(DatasetAttackResult):
    positive_probabilities: np.ndarray
    negative_probabilities: np.ndarray


@dataclass
class DatasetAttackScore:
    dataset_name: str
