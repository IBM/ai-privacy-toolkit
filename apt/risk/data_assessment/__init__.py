"""
Module providing privacy risk assessment for synthetic data.

The main interface, ``DatasetAttack``, with the ``assess_privacy()`` main method assumes the availability of the
training data, holdout data and synthetic data at the time of the privacy evaluation.
It is to be implemented by concrete assessment methods, which can run the assessment on a per-record level,
or on the whole dataset.
The abstract class ``DatasetAttackMembership`` implements the ``DatasetAttack`` interface, but adds the result
of the membership inference attack, so that the final score contains both the membership inference attack result
for further analysis and the calculated score.
"""
from apt.risk.data_assessment import dataset_attack
