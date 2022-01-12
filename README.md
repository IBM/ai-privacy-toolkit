# ai-privacy-toolkit
<p align="center">
  <img src="docs/images/logo with text.jpg?raw=true" width="467" title="ai-privacy-toolkit logo">
</p>
<br />

A toolkit for tools and techniques related to the privacy and compliance of AI models.

The [**anonymization**](apt/anonymization/README.md) module contains methods for anonymizing ML model 
training data, so that when a model is retrained on the anonymized data, the model itself will also be 
considered anonymous. This may help exempt the model from different obligations and restrictions 
set out in data protection regulations such as GDPR, CCPA, etc. 

The [**minimization**](apt/minimization/README.md) module contains methods to help adhere to the data 
minimization principle in GDPR for ML models. It enables to reduce the amount of 
personal data needed to perform predictions with a machine learning model, while still enabling the model
to make accurate predictions. This is done by by removing or generalizing some of the input features.

Official ai-privacy-toolkit documentation: https://ai-privacy-toolkit.readthedocs.io/en/latest/

Installation: pip install ai-privacy-toolkit

For more information or help using or improving the toolkit, please contact Abigail Goldsteen at abigailt@il.ibm.com, 
or join our Slack channel: https://aip360.mybluemix.net/community.

**Related toolkits:**

ai-minimization-toolkit - has been migrated into this toolkit.

[differential-privacy-library](https://github.com/IBM/differential-privacy-library): A 
general-purpose library for experimenting with, investigating and developing applications in, 
differential privacy.

[adversarial-robustness-toolbox](https://github.com/Trusted-AI/adversarial-robustness-toolbox):
A Python library for Machine Learning Security. Includes an attack module called *inference* that contains privacy attacks on ML models 
(membership inference, attribute inference, model inversion and database reconstruction) as well as a *privacy* metrics module that contains
membership leakage metrics for ML models.

