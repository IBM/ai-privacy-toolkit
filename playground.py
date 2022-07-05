from apt.minimization.orderedfeatureminimizer import OrderedFeatureMinimizer
from data_minimization_benchmark.datapreparers import GSSDataPreparer
from sklearn.ensemble import RandomForestClassifier
from data_minimization_benchmark.utils.NormalizedCertaintyPenalty import calculate_normalized_certainty_penalty
from apt.minimization import ShapMinimizer

if __name__ == '__main__':

    ordered_shap_features = ['Race', 'Gender', 'X_rated', 'Children', 'Work status', 'Age', 'Happiness']

    gss_data = GSSDataPreparer().prepare()
    encoded_data, unencoded_data = gss_data.encoded_data, gss_data.unencoded_data
    random_forest = RandomForestClassifier()
    random_forest.fit(*encoded_data.model_train)
    # minimizer = ShapMinimizer(random_forest, categorical_features=gss_data.categorical_features, target_accuracy=0.8,
    #                          background_size=20, n_samples=70)
    minimizer2 = OrderedFeatureMinimizer(random_forest, categorical_features=gss_data.categorical_features,
                                         target_accuracy=0.8, ordered_features=ordered_shap_features)
    minimizer2.fit(unencoded_data.model_train.X)
    ret = minimizer2.transform(unencoded_data.model_train.X)

