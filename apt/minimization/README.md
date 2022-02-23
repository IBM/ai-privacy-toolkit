# data minimization module

The EU General Data Protection Regulation (GDPR) mandates the principle of data minimization, which requires that only 
data necessary to fulfill a certain purpose be collected. However, it can often be difficult to determine the minimal 
amount of data required, especially in complex machine learning models such as neural networks. 

This module implements a first-of-a-kind method to help reduce the amount of personal data needed to perform 
predictions with a machine learning model, by removing or generalizing some of the input features. The type of data 
minimization this toolkit focuses on is the reduction of the number and/or granularity of features collected for analysis. 

The generalization process basically searches for several similar records and groups them together. Then, for each 
feature, the individual values for that feature within each group are replaced with a represenataive value that is 
common across the whole group. This process is done while using knowledge encoded within the model to produce a 
generalization that has little to no impact on its accuracy. 

For more information about the method see: http://export.arxiv.org/pdf/2008.04113

The following figure depicts the overall process:

<p align="center">
  <img src="../../docs/images/AI_Privacy_project.jpeg?raw=true" width="667" title="data minimization process">
</p>
<br />

Usage
-----

The main class, ``GeneralizeToRepresentative``, is a scikit-learn compatible ``Transformer``, that receives an existing 
estimator and labeled training data, and learns the generalizations that can be applied to any newly collected data for 
analysis by the original model. The ``fit()`` method learns the generalizations and the ``transform()`` method applies 
them to new data.

It is also possible to export the generalizations as feature ranges.

The current implementation supports numeric features and categorical features.

Start by training your machine learning model. In this example, we will use a ``DecisionTreeClassifier``, but any 
scikit-learn model can be used. We will use the iris dataset in our example.

```
  from sklearn import datasets
  from sklearn.model_selection import train_test_split
  from sklearn.tree import DecisionTreeClassifier

  dataset = datasets.load_iris()
  X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.2)

  base_est = DecisionTreeClassifier()
  base_est.fit(X_train, y_train)
```

Now create the ``GeneralizeToRepresentative`` transformer and train it. Supply it with the original model and the 
desired target accuracy. The training process may receive the original labeled training data or the model's predictions 
on the data.

```
  predictions = base_est.predict(X_train)
  gen = GeneralizeToRepresentative(base_est, target_accuracy=0.9)
  gen.fit(X_train, predictions)
```

Now use the transformer to transform new data, for example the test data.

```
  transformed = gen.transform(X_test)
```

The transformed data has the same columns and formats as the original data, so it can be used directly to derive 
predictions from the original model.

```
  new_predictions = base_est.predict(transformed)
```

To export the resulting generalizations, retrieve the ``Transformer``'s ``_generalize`` parameter.

```
  generalizations = base_est._generalize
```

The returned object has the following structure::

  {
    ranges: 
    {
      list of (<feature name>: [<list of values>])
    }, 
    untouched: [<list of feature names>]
  }
  
For example::

  {
    ranges: 
    {
      age: [21.5, 39.0, 51.0, 70.5], 
      education-years: [8.0, 12.0, 14.5]
    }, 
    untouched: ["occupation", "marital-status"]
  }
  
Where each value inside the range list represents a cutoff point. For example, for the ``age`` feature, the ranges in 
this example are: ``<21.5, 21.5-39.0, 39.0-51.0, 51.0-70.5, >70.5``. The ``untouched`` list represents features that 
were not generalized, i.e., their values should remain unchanged.

Citation
--------
Goldsteen, A., Ezov, G., Shmelkin, R. et al. Data minimization for GDPR compliance in machine learning models. AI Ethics 
(2021). https://doi.org/10.1007/s43681-021-00095-8




