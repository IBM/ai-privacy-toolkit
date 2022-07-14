import numpy as np

from apt.minimization.orderedfeatureminimizer import OrderedFeatureMinimizer
from data_minimization_benchmark.datapreparers import GSSDataPreparer
from sklearn.ensemble import RandomForestClassifier
from data_minimization_benchmark.utils.NormalizedCertaintyPenalty import calculate_normalized_certainty_penalty
from apt.minimization import ShapMinimizer
from sklearn.metrics import accuracy_score
from data_minimization_benchmark.benchmark import Benchmark
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Previously calculated order of shap values
    # ordered_shap_features = ['Race', 'Gender', 'X_rated', 'Children', 'Work status', 'Age', 'Happiness']
    ordered_shap_features = ['Race', 'Gender', 'X_rated', 'Children', 'Age', 'Work status', 'Happiness']

    gss_data = GSSDataPreparer().prepare()
    encoded_data, unencoded_data = gss_data.encoded_data, gss_data.unencoded_data
    random_forest = RandomForestClassifier()
    random_forest.fit(*encoded_data.model_train)
    # minimizer = ShapMinimizer(random_forest, gss_data._encoder, categorical_features=gss_data.categorical_features, target_accuracy=0.8,
    #                          background_size=20, n_samples=70)
    minimizer = OrderedFeatureMinimizer(random_forest, data_encoder=gss_data._encoder,
                                        categorical_features=gss_data.categorical_features,
                                        target_accuracy=0.8, ordered_features=ordered_shap_features)
    minimizer.fit(unencoded_data.model_train.X)
    X_transformed = minimizer.transform(unencoded_data.model_train.X)
    # print(
    #     f"Relative accuracy = {accuracy_score(random_forest.predict(X_transformed), random_forest.predict(encoded_data.model_train.X))}")
    # print(minimizer.generalizations)


    def shap_minimizer_maker(*args, **kwargs):
        return OrderedFeatureMinimizer(*args, **kwargs, ordered_features=ordered_shap_features,
                                       data_encoder=gss_data._encoder)

    target_accuracies = np.arange(0
                                  , 1.0000001, 0.1)
    benchmark = Benchmark(
        # generalizer=OrderedFeatureMinimizer(random_forest, data_encoder=gss_data._encoder,
        #                                 categorical_features=gss_data.categorical_features,
        #                                 target_accuracy=0.8, ordered_features=ordered_shap_features),
        generalizer=shap_minimizer_maker,
        train_set=gss_data.unencoded_data.generalizer_train,
        test_set=gss_data.unencoded_data.generalizer_validation,
        models={"random_forest": random_forest},
        transformer=gss_data._encoder,
        target_accuracies=target_accuracies,
        features=gss_data.all_features,
        categorical_features=gss_data.categorical_features,
    )
    res = benchmark.run()
    plt.title("GSS Dataset - Random Forest")
    plt.gca().invert_xaxis()
    plt.xlabel("Target Accuracies")
    plt.ylabel("NCP")
    plt.plot(target_accuracies, [x.ncp for x in res["random_forest"]])
    plt.show()
    plt.title("GSS Dataset - Random Forest")
    plt.gca().invert_xaxis()
    plt.xlabel("Target Accuracies")
    plt.ylabel("Relative Accuracy (test set)")
    plt.plot(target_accuracies, [x.test_relative_accuracy for x in res["random_forest"]])
    plt.show()
