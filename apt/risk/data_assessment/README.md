# Privacy Assessment of Datasets for AI Models

This module implements a tool for privacy assessment of synthetic datasets that are to be used in AI model training.

The main interface, ``DatasetAttack``, with the ``assess_privacy()`` main method assumes the availability of the
training data, holdout data and synthetic data at the time of the privacy evaluation.
It is to be implemented by concrete assessment methods, which can run the assessment on a per-record level,
or on the whole dataset.
The method ``assess_privacy()`` returns a ``DatasetAttackScore``, which contains a ``risk_score`` and,
optionally, a ``DatasetAttackResult``. Each specific attack can implement its own ``DatasetAttackScore``, which would
contain additional fields.

The abstract class ``DatasetAttackMembership`` implements the ``DatasetAttack`` interface, but adds the result
of the membership inference attack, so that the final score contains both the membership inference attack result
for further analysis and the calculated score.


``DatasetAssessmentManager`` provides convenience methods to run multiple attacks and persist the result reports.

Attack Implementations
-----------------------

One implementation is based on the paper "GAN-Leaks: A Taxonomy of Membership Inference Attacks against Generative
Models"[^1] and its implementation[^2]. It is based on Black-Box MIA attack using
distances of members (training set) and non-members (holdout set) from their nearest neighbors in the synthetic dataset.
By default, the Euclidean distance is used (L2 norm), but another ``compute_distance()`` method can be provided in
configuration instead.
The area under the receiver operating characteristic curve (AUC ROC) gives the privacy risk score.

Another implementation is based on the papers "Data Synthesis based on Generative Adversarial Networks"[^3] and
"Holdout-Based Fidelity and Privacy Assessment of Mixed-Type Synthetic Data"[^4], and on a variation of its reference
implementation[^5].
It is based on distances of synthetic data records from members (training set) and non-members (holdout set).
The privacy risk score is the share of synthetic records closer to the training than the holdout dataset.
By default, the Euclidean distance is used (L2 norm), but another ``compute_distance()`` method can be provided in
configuration instead.

Usage
-----
An implementation of the ``DatasetAttack`` interface is used for performing a privacy attack for risk assessment of
synthetic datasets to be used in AI model training.
The original data members (training data), non-members (the holdout data) and the synthetic data created from the
original members should be available.
For reliability, all the datasets should be preprocessed and normalized.

The following example runs all the attacks and persists the results in files, using ``DatasetAssessmentManager``.
It assumes that you provide it with the pairs ``(x_train, y_train)``, ``(x_test, y_test)`` and ``(x_synth, y_synth)``
for members, non-members and the synthetic datasets, respectively.

```python
from apt.risk.data_assessment.dataset_assessment_manager import DatasetAssessmentManager, \
    DatasetAssessmentManagerConfig
from apt.utils.datasets import ArrayDataset

dataset_assessment_manager = DatasetAssessmentManager(
    DatasetAssessmentManagerConfig(persist_reports=True, generate_plots=False))

synthetic_data = ArrayDataset(x_synth, y_synth)
original_data_members = ArrayDataset(x_train, y_train)
original_data_non_members = ArrayDataset(x_test, y_test)

dataset_name = 'my_dataset'
[score_gl, score_h] = dataset_assessment_manager.assess(
    original_data_members, original_data_non_members, synthetic_data, dataset_name)
dataset_assessment_manager.dump_all_scores_to_files()
```

Alternatively, each attack can be run separately, for instance:

```python
from apt.risk.data_assessment.dataset_attack_membership_knn_probabilities import \
    DatasetAttackConfigMembershipKnnProbabilities, DatasetAttackMembershipKnnProbabilities
from apt.utils.datasets import ArrayDataset

synthetic_data = ArrayDataset(x_synth, y_synth)
original_data_members = ArrayDataset(x_train, y_train)
original_data_non_members = ArrayDataset(x_test, y_test)

config_gl = DatasetAttackConfigMembershipKnnProbabilities(use_batches=False,
                                                          generate_plot=False)
attack_gl = DatasetAttackMembershipKnnProbabilities(original_data_members,
                                                    original_data_non_members,
                                                    synthetic_data,
                                                    config_gl)

score_gl = attack_gl.assess_privacy()
```

Citations
---------

  [^1]: "GAN-Leaks: A Taxonomy of Membership Inference Attacks against Generative Models" by D. Chen, N. Yu, Y. Zhang,
    M. Fritz in Proceedings of the 2020 ACM SIGSAC Conference on Computer and Communications Security, 343–62, 2020.
    [https://doi.org/10.1145/3372297.3417238](https://doi.org/10.1145/3372297.3417238)

  [^2]: Code for the paper "GAN-Leaks: A Taxonomy of Membership Inference Attacks against Generative Models"
    [https://github.com/DingfanChen/GAN-Leaks](https://github.com/DingfanChen/GAN-Leaks)

  [^3]: "Data Synthesis based on Generative Adversarial Networks." by N. Park, M. Mohammadi, K. Gorde, S. Jajodia,
    H. Park, and Y. Kim in International Conference on Very Large Data Bases (VLDB), 2018.

  [^4]: "Holdout-Based Fidelity and Privacy Assessment of Mixed-Type Synthetic Data" by M. Platzer and T. Reutterer.

  [^5]: Code for the paper "Holdout-Based Fidelity and Privacy Assessment of Mixed-Type Synthetic Data"
    [https://github.com/mostly-ai/paper-fidelity-accuracy](https://github.com/mostly-ai/paper-fidelity-accuracy)
