.. ai-privacy-toolkit documentation master file, created by
   sphinx-quickstart on Mon Feb 15 12:42:20 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to ai-privacy-toolkit's documentation!
==============================================

This project provides tools for assessing and improving the privacy and compliance of AI models.

The first release of this toolkit contains a single module called anonymization. This
module contains methods for anonymizing ML model training data, so that when
a model is retrained on the anonymized data, the model itself will also be considered
anonymous. This may help exempt the model from different obligations and restrictions
set out in data protection regulations such as GDPR, CCPA, etc.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started:

   quick_start

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: API

   apt



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
