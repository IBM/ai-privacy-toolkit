# ai-privacy-toolkit
<p align="center">
  <img src="docs/images/logo with text.jpg?raw=true" width="467" title="ai-privacy-toolkit logo">
</p>
<br />

A toolkit for tools and techniques related to the privacy and compliance of AI models.

The first release of this toolkit contains a single module called [**anonymization**](apt/anonymization/README.md).
This module contains methods for anonymizing ML model training data, so that when 
a model is retrained on the anonymized data, the model itself will also be considered 
anonymous. This may help exempt the model from different obligations and restrictions 
set out in data protection regulations such as GDPR, CCPA, etc. 

Official ai-privacy-toolkit documentation: https://ai-privacy-toolkit.readthedocs.io/en/latest/

**Related toolkits:**

[ai-minimization-toolkit](https://github.com/IBM/ai-minimization-toolkit): A toolkit for 
reducing the amount of personal data needed to perform predictions with a machine learning model

[differential-privacy-library](https://github.com/IBM/differential-privacy-library): A 
general-purpose library for experimenting with, investigating and developing applications in, 
differential privacy.

[adversarial-robustness-toolbox](https://github.com/Trusted-AI/adversarial-robustness-toolbox):
A Python library for Machine Learning Security.

