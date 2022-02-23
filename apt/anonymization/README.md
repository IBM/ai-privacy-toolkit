# anonymization module
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

The following figure depicts the overall process:

<p align="center">
  <img src="../../docs/images/AI_Privacy_project2.jpg?raw=true" width="667" title="anonymization process">
</p>
<br />

Citation
--------
Goldsteen A., Ezov G., Shmelkin R., Moffie M., Farkash A. (2022) Anonymizing Machine Learning Models. In: Garcia-Alfaro 
J., Muñoz-Tapia J.L., Navarro-Arribas G., Soriano M. (eds) Data Privacy Management, Cryptocurrencies and Blockchain 
Technology. DPM 2021, CBT 2021. Lecture Notes in Computer Science, vol 13140. Springer, Cham. 
https://doi.org/10.1007/978-3-030-93944-1_8


