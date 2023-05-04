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

The [**dataset assessment**](apt/risk/data_assessment/README.md) module implements a tool for privacy assessment of
synthetic datasets that are to be used in AI model training.

Official ai-privacy-toolkit documentation: https://ai-privacy-toolkit.readthedocs.io/en/latest/

Installation: pip install ai-privacy-toolkit

For more information or help using or improving the toolkit, please contact Abigail Goldsteen at abigailt@il.ibm.com, 
or join our Slack channel: https://aip360.mybluemix.net/community.

We welcome new contributors! If you're interested, take a look at our [**contribution guidelines**](https://github.com/IBM/ai-privacy-toolkit/wiki/Contributing).

**Related toolkits:**

ai-minimization-toolkit - has been migrated into this toolkit.

[differential-privacy-library](https://github.com/IBM/differential-privacy-library): A 
general-purpose library for experimenting with, investigating and developing applications in, 
differential privacy.

[adversarial-robustness-toolbox](https://github.com/Trusted-AI/adversarial-robustness-toolbox):
A Python library for Machine Learning Security. Includes an attack module called *inference* that contains privacy attacks on ML models 
(membership inference, attribute inference, model inversion and database reconstruction) as well as a *privacy* metrics module that contains
membership leakage metrics for ML models.


Citation
--------
Abigail Goldsteen, Ola Saadi, Ron Shmelkin, Shlomit Shachor, Natalia Razinkov,
"AI privacy toolkit", SoftwareX, Volume 22, 2023, 101352, ISSN 2352-7110, https://doi.org/10.1016/j.softx.2023.101352.