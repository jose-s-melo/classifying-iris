import random_search

from sklearn.neural_network import MLPClassifier


random_search_results: random_search.GetParamsReturn = random_search.get_params()
random_search_best_params: random_search.MLPParams = random_search_results["best_params"]

mlp = MLPClassifier(
    hidden_layer_sizes=(random_search_best_params['hidden_neurons_size'],),
    activation=random_search_best_params['activation'],
    alpha=random_search_best_params['alpha'],
    solver=random_search_best_params['solver'],
    max_iter=random_search_best_params['max_iterations']
)