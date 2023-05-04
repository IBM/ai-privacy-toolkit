.. ai-privacy-toolkit documentation master file, created by
   sphinx-quickstart on Mon Feb 15 12:42:20 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to ai-privacy-toolkit's documentation!
==============================================

This project provides tools for assessing and improving the privacy and compliance of AI models.

The anonymization module contains methods for anonymizing ML model
training data, so that when a model is retrained on the anonymized data, the model itself will also be
considered anonymous. This may help exempt the model from different obligations and restrictions
set out in data protection regulations such as GDPR, CCPA, etc.

The minimization module contains methods to help adhere to the data
minimization principle in GDPR for ML models. It enables to reduce the amount of
personal data needed to perform predictions with a machine learning model, while still enabling the model
to make accurate predictions. This is done by by removing or generalizing some of the input features.

The dataset risk assessment module implements a tool for privacy assessment of synthetic datasets that are to be used in AI model training.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started:

   source/quick_start

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: API

   source/apt



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
