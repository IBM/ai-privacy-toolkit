"""
Module providing privacy risk assessment for synthetic data.

The main interface, ``DatasetAttack``, with the assess_privacy() main method assumes the availability of the
training data, holdout data and synthetic data at the time of the privacy evaluation.
It is implemented by two types of abstract classes: ``DatasetAttackPerRecord`` and ``DatasetAttackWhole``, to be
implemented by concrete assessment methods.
"""
from apt.risk.data_assessment import dataset_attack
