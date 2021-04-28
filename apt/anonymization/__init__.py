"""
Module providing ML anonymization.

This module contains methods for anonymizing ML model training data, so that when
a model is retrained on the anonymized data, the model itself will also be considered
anonymous. This may help exempt the model from different obligations and restrictions
set out in data protection regulations such as GDPR, CCPA, etc.

The module contains methods that enable anonymizing training datasets in a manner that
is tailored to and guided by an existing, trained ML model. It uses the existing model's
predictions on the training data to train a second, anonymizer model, that eventually determines
the generalizations that will be applied to the training data. For more information about the
method see: https://arxiv.org/abs/2007.13086

Once the anonymized training data is returned, it can be used to retrain the model.
"""
from apt.anonymization.anonymizer import Anonymize
