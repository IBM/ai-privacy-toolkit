import pandas as pd
import pytest

from apt.anonymization import Anonymize
from apt.risk.data_assessment.dataset_assessment_manager import DatasetAssessmentManager, DatasetAssessmentManagerConfig
from apt.utils.dataset_utils import get_iris_dataset_np, get_nursery_dataset_pd
from apt.utils.datasets import ArrayDataset
from apt.risk.data_assessment.dataset_attack_membership_classification import DatasetAttackScoreMembershipClassification
from apt.risk.data_assessment.dataset_attack_membership_knn_probabilities import \
    DatasetAttackScoreMembershipKnnProbabilities, DatasetAttackConfigMembershipKnnProbabilities, \
    DatasetAttackMembershipKnnProbabilities
from apt.risk.data_assessment.dataset_attack_whole_dataset_knn_distance import DatasetAttackScoreWholeDatasetKnnDistance
from tests.test_data_assessment import kde, preprocess_nursery_x_data

NUM_SYNTH_SAMPLES = 10
NUM_SYNTH_COMPONENTS = 2
ANON_K = 2
MIN_SHARE = 0.5
MIN_ROC_AUC = 0.0
MIN_PRECISION = 0.0

iris_dataset_np = get_iris_dataset_np()
nursery_dataset_pd = get_nursery_dataset_pd()

mgr1 = DatasetAssessmentManager(DatasetAssessmentManagerConfig(persist_reports=False, generate_plots=False))
mgr2 = DatasetAssessmentManager(DatasetAssessmentManagerConfig(persist_reports=False, generate_plots=True))
mgr3 = DatasetAssessmentManager(DatasetAssessmentManagerConfig(persist_reports=True, generate_plots=False))
mgr4 = DatasetAssessmentManager(DatasetAssessmentManagerConfig(persist_reports=True, generate_plots=True))
mgrs = [mgr1, mgr2, mgr3, mgr4]


def teardown_function():
    for mgr in mgrs:
        mgr.dump_all_scores_to_files()


anon_testdata = ([('iris_np', iris_dataset_np, 'np', mgr1)]
                 + [('nursery_pd', nursery_dataset_pd, 'pd', mgr2)]
                 + [('iris_np', iris_dataset_np, 'np', mgr3)]
                 + [('nursery_pd', nursery_dataset_pd, 'pd', mgr4)])


@pytest.mark.parametrize("name, data, dataset_type, mgr", anon_testdata)
def test_risk_anonymization(name, data, dataset_type, mgr):
    (x_train, y_train), (x_test, y_test) = data

    if dataset_type == 'np':
        # no need to preprocess
        preprocessed_x_train = x_train
        preprocessed_x_test = x_test
        QI = [0, 2]
        anonymizer = Anonymize(ANON_K, QI, train_only_QI=True)
        categorical_features = []
    elif "nursery" in name:
        preprocessed_x_train, preprocessed_x_test, categorical_features = preprocess_nursery_x_data(x_train, x_test)
        preprocessed_x_train = pd.DataFrame(preprocessed_x_train)
        preprocessed_x_test = pd.DataFrame(preprocessed_x_test)
        QI = list(range(15, 20))
        anonymizer = Anonymize(ANON_K, QI, train_only_QI=True)
    else:
        raise ValueError('Pandas dataset missing a preprocessing step')

    anonymized_data = ArrayDataset(anonymizer.anonymize(ArrayDataset(preprocessed_x_train, y_train)))
    original_data_members = ArrayDataset(preprocessed_x_train, y_train)
    original_data_non_members = ArrayDataset(preprocessed_x_test, y_test)

    dataset_name = f'anon_k{ANON_K}_{name}'
    assess_privacy_and_validate_result(mgr, original_data_members, original_data_non_members, anonymized_data,
                                       dataset_name, categorical_features)

    assess_privacy_and_validate_result(mgr, original_data_members=original_data_members,
                                       original_data_non_members=original_data_non_members,
                                       synth_data=anonymized_data, dataset_name=None,
                                       categorical_features=categorical_features)


testdata = [('iris_np', iris_dataset_np, 'np', mgr4),
            ('nursery_pd', nursery_dataset_pd, 'pd', mgr3),
            ('iris_np', iris_dataset_np, 'np', mgr2),
            ('nursery_pd', nursery_dataset_pd, 'pd', mgr1)]


@pytest.mark.parametrize("name, data, dataset_type, mgr", testdata)
def test_risk_kde(name, data, dataset_type, mgr):
    original_data_members, original_data_non_members, synthetic_data, categorical_features \
        = encode_and_generate_synthetic_data(dataset_type, name, data)

    dataset_name = 'kde' + str(NUM_SYNTH_SAMPLES) + name
    assess_privacy_and_validate_result(mgr, original_data_members, original_data_non_members, synthetic_data,
                                       dataset_name, categorical_features)

    assess_privacy_and_validate_result(mgr, original_data_members=original_data_members,
                                       original_data_non_members=original_data_non_members,
                                       synth_data=synthetic_data, dataset_name=None,
                                       categorical_features=categorical_features)


testdata_knn_options = [('iris_np', iris_dataset_np, 'np'),
                        ('nursery_pd', nursery_dataset_pd, 'pd')]


@pytest.mark.parametrize("name, data, dataset_type", testdata_knn_options)
def test_risk_kde_knn_options(name, data, dataset_type):
    original_data_members, original_data_non_members, synthetic_data, categorical_features \
        = encode_and_generate_synthetic_data(dataset_type, name, data)

    dataset_name = 'kde' + str(NUM_SYNTH_SAMPLES) + name

    config_g = DatasetAttackConfigMembershipKnnProbabilities(use_batches=True, generate_plot=False,
                                                             distribution_comparison_alpha=0.1)
    numeric_tests = ['KS', 'CVM', 'AD', 'ES']
    categorical_tests = ['CHI', 'AD', 'ES']
    for numeric_test in numeric_tests:
        for categorical_test in categorical_tests:
            attack_g = DatasetAttackMembershipKnnProbabilities(original_data_members,
                                                               original_data_non_members,
                                                               synthetic_data,
                                                               config_g,
                                                               dataset_name,
                                                               categorical_features,
                                                               distribution_comparison_numeric_test=numeric_test,
                                                               distribution_comparison_categorical_test=categorical_test
                                                               )

            score_g = attack_g.assess_privacy()
            assert score_g.roc_auc_score > MIN_ROC_AUC
            assert score_g.average_precision_score > MIN_PRECISION


def encode_and_generate_synthetic_data(dataset_type, name, data):
    (x_train, y_train), (x_test, y_test) = data

    if dataset_type == 'np':
        encoded = x_train
        encoded_test = x_test
        num_synth_components = NUM_SYNTH_COMPONENTS
        categorical_features = []
    elif "nursery" in name:
        encoded, encoded_test, categorical_features = preprocess_nursery_x_data(x_train, x_test)
        num_synth_components = 10
    else:
        raise ValueError('Pandas dataset missing a preprocessing step')
    synthetic_data = ArrayDataset(
        kde(NUM_SYNTH_SAMPLES, n_components=num_synth_components, original_data=encoded))
    original_data_members = ArrayDataset(encoded, y_train)
    original_data_non_members = ArrayDataset(encoded_test, y_test)
    return original_data_members, original_data_non_members, synthetic_data, categorical_features


def assess_privacy_and_validate_result(mgr, original_data_members, original_data_non_members, synth_data, dataset_name,
                                       categorical_features):
    attack_scores = mgr.assess(original_data_members, original_data_non_members, synth_data, dataset_name,
                               categorical_features)

    for i, (assessment_type, scores) in enumerate(attack_scores.items()):
        if assessment_type == 'MembershipKnnProbabilities':
            score_g: DatasetAttackScoreMembershipKnnProbabilities = scores[0]
            assert score_g.roc_auc_score > MIN_ROC_AUC
            assert score_g.average_precision_score > MIN_PRECISION
        elif assessment_type == 'WholeDatasetKnnDistance':
            score_h: DatasetAttackScoreWholeDatasetKnnDistance = scores[0]
            assert score_h.share > MIN_SHARE
        if assessment_type == 'MembershipClassification':
            score_mc: DatasetAttackScoreMembershipClassification = scores[0]
            assert score_mc.synthetic_data_quality_warning is False
            assert 0 <= score_mc.normalized_ratio <= 1
