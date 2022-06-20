from apt.minimization.shapminimizer import Minimizer
from data_minimization_benchmark.datapreparers import GSSDataPreparer
from sklearn.ensemble import RandomForestClassifier
from data_minimization_benchmark.utils.NormalizedCertaintyPenalty import calculate_normalized_certainty_penalty

if __name__ == '__main__':
    gss_data = GSSDataPreparer().prepare()
    encoded_data, unencoded_data = gss_data.encoded_data, gss_data.unencoded_data
    random_forest = RandomForestClassifier()
    random_forest.fit(*encoded_data.model_train)
    minimizer = Minimizer(random_forest, categorical_features=gss_data.categorical_features)
    ret = minimizer.fit(unencoded_data.model_train.X)

