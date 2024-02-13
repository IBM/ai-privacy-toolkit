from dataclasses import dataclass
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from apt.risk.data_assessment.dataset_attack import DatasetAttackMembership, Config
from apt.risk.data_assessment.dataset_attack_result import DatasetAttackScore, DEFAULT_DATASET_NAME
from apt.utils.datasets import ArrayDataset


@dataclass
class DatasetAttackConfigMembershipClassification(Config):
    """Configuration for DatasetAttackMembershipClassification.

    Attributes:
        classifier_type:  sklearn classifier type for the member classification.
                          Can be LogisticRegression or RandomForestClassifier
        threshold:  a minimum threshold of distinguishability, above which a synthetic_data_quality_warning is raised.
                    A value higher than the threshold means that it is too easy to distinguish between the synthetic
                    data and the training or test data.
    """
    classifier_type: str = 'RandomForestClassifier'
    threshold: float = 0.9


@dataclass
class DatasetAttackScoreMembershipClassification(DatasetAttackScore):
    """DatasetAttackMembershipClassification privacy risk score.
    """
    member_roc_auc_score: float
    non_member_roc_auc_score: float
    normalized_ratio: float
    synthetic_data_quality_warning: bool
    assessment_type: str = 'MembershipClassification'  # to be used in reports

    def __init__(self, dataset_name: str, member_roc_auc_score: float, non_member_roc_auc_score: float,
                 normalized_ratio: float, synthetic_data_quality_warning: bool) -> None:
        """
        dataset_name:    dataset name to be used in reports
        member_roc_auc_score:  ROC AUC score of classification between members (training) data and synthetic data
        non_member_roc_auc_score: ROC AUC score of classification between non-members (test) data and synthetic data,
                                    this is the baseline score
        normalized_ratio:  ratio of the member_roc_auc_score to the non_member_roc_auc_score
        synthetic_data_quality_warning:  True if either the member_roc_auc_score or the non_member_roc_auc_score is
                                        higher than the threshold. That means that the synthetic data does not represent
                                        the training data sufficiently well, or that the test data is too far from the
                                        synthetic data.
        """
        super().__init__(dataset_name=dataset_name, risk_score=normalized_ratio, result=None)
        self.member_roc_auc_score = member_roc_auc_score
        self.non_member_roc_auc_score = non_member_roc_auc_score
        self.normalized_ratio = normalized_ratio
        self.synthetic_data_quality_warning = synthetic_data_quality_warning


class DatasetAttackMembershipClassification(DatasetAttackMembership):
    """
         Privacy risk assessment for synthetic datasets that compares the distinguishability of the synthetic dataset
         from the members dataset (training) as opposed to the distinguishability of the synthetic dataset from the
         non-members dataset (test).
         The privacy risk measure is calculated as the ratio of the receiver operating characteristic curve (AUC ROC) of
         the members dataset to AUC ROC of the non-members dataset. It can be 0.0 or higher, with higher scores meaning
         higher privacy risk and worse privacy.
    """
    SHORT_NAME = 'MembershipClassification'

    def __init__(self, original_data_members: ArrayDataset, original_data_non_members: ArrayDataset,
                 synthetic_data: ArrayDataset,
                 config: DatasetAttackConfigMembershipClassification = DatasetAttackConfigMembershipClassification(),
                 dataset_name: str = DEFAULT_DATASET_NAME, categorical_features: list = None):
        """
        :param original_data_members: A container for the training original samples and labels. Should be encoded and
                                      scaled.
        :param original_data_non_members: A container for the holdout original samples and labels. Should be encoded
                                          and scaled.
        :param synthetic_data: A container for the synthetic samples and labels. Should be encoded and scaled.
        :param config: Configuration parameters to guide the attack, optional
        :param dataset_name: A name to identify this dataset, optional
        """
        super().__init__(original_data_members, original_data_non_members, synthetic_data, config, dataset_name,
                         categorical_features)
        self.member_classifier = self._get_classifier(config.classifier_type)
        self.non_member_classifier = self._get_classifier(config.classifier_type)
        self.threshold = config.threshold

    def short_name(self):
        return self.SHORT_NAME

    @staticmethod
    def _get_classifier(classifier_type):
        if classifier_type == 'LogisticRegression':
            classifier = LogisticRegression()
        elif classifier_type == 'RandomForestClassifier':
            classifier = RandomForestClassifier(max_depth=2, random_state=0)
        else:
            raise ValueError('Incorrect classifier type', classifier_type)
        return classifier

    def assess_privacy(self) -> DatasetAttackScoreMembershipClassification:
        """
        Calculate the ratio of the receiver operating characteristic curve (AUC ROC) of the distinguishability of the
        synthetic data from the members dataset to AU ROC of the distinguishability of the synthetic data from the
        non-members dataset.
        :return: the ratio as the privacy risk measure
        """
        member_roc_auc = self._classify_datasets(
            self.original_data_members, self.synthetic_data, self.member_classifier)
        non_member_roc_auc = self._classify_datasets(
            self.original_data_non_members, self.synthetic_data, self.non_member_classifier)

        score = self.calculate_privacy_score(member_roc_auc, non_member_roc_auc)
        return score

    def _classify_datasets(self, df1: ArrayDataset, df2: ArrayDataset, classifier):
        """
        Split df1 and df2 into train and test parts, fit the classifier to distinguish between df1 train and
        df2 train, and then check how good the classification is on the df1 test and df2 test parts.
        :return: ROC AUC score of the classification between df1 test and df2 test
        """
        df1_train, df1_test = train_test_split(df1.get_samples(), test_size=0.5, random_state=42)

        df2_train, df2_test = train_test_split(df2.get_samples(), test_size=0.5, random_state=42)

        train_x = np.concatenate([df1_train, df2_train])
        train_labels = np.concatenate((np.ones_like(df1_train[:, 0], dtype='int'),
                                       np.zeros_like(df2_train[:, 0], dtype='int')))

        classifier.fit(train_x, train_labels)

        test_x = np.concatenate([df1_test, df2_test])
        test_labels = np.concatenate((np.ones_like(df1_test[:, 0], dtype='int'),
                                      np.zeros_like(df2_test[:, 0], dtype='int')))

        print('Model accuracy: ', classifier.score(test_x, test_labels))
        predict_proba = classifier.predict_proba(test_x)
        return roc_auc_score(test_labels, predict_proba[:, 1])

    def calculate_privacy_score(self, member_roc_auc: float, non_member_roc_auc: float) -> (
            DatasetAttackScoreMembershipClassification):
        """
        Compare the distinguishability of the synthetic data from the members dataset (training)
         with the distinguishability of the synthetic data from the non-members dataset (test).
        :return:
        """
        score, baseline_score = member_roc_auc, non_member_roc_auc

        if 0 < baseline_score <= score:
            normalized_ratio = score / baseline_score - 1.0
        else:
            normalized_ratio = 0

        if (score >= self.threshold) or (baseline_score >= self.threshold):
            synthetic_data_quality_warning = True
        else:
            synthetic_data_quality_warning = False

        score = DatasetAttackScoreMembershipClassification(
            self.dataset_name, member_roc_auc_score=score, non_member_roc_auc_score=baseline_score,
            normalized_ratio=normalized_ratio, synthetic_data_quality_warning=synthetic_data_quality_warning)

        return score
