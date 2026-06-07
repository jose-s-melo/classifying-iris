import random

from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import RobustScaler

from mlp.common import (
    MLPParams,
    HistoricItem,
    GetParamsReturn
)

dataset = load_iris()

X = dataset.data
Y = dataset.target

X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.3,
    stratify=Y,
    random_state=50
)

scaler = RobustScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

def create_individual(
    hidden_neurons: list[int],
    activations: list[str],
    alphas: list[float],
    solvers: list[str],
    num_iterations: list[int]
) -> MLPParams:

    return {
        "hidden_neurons_size": random.choice(hidden_neurons),
        "activation": random.choice(activations),
        "alpha": random.choice(alphas),
        "solver": random.choice(solvers),
        "max_iterations": random.choice(num_iterations)
    }


def evaluate(individual: MLPParams) -> float:

    mlp = MLPClassifier(
        hidden_layer_sizes=(
            individual["hidden_neurons_size"],
        ),
        activation=individual["activation"],
        solver=individual["solver"],
        alpha=individual["alpha"],
        max_iter=individual["max_iterations"],
        random_state=50
    )

    mlp.fit(X_train, Y_train)

    predictions = mlp.predict(X_test)

    return accuracy_score(
        Y_test,
        predictions
    )


def crossover(
    parent1: MLPParams,
    parent2: MLPParams
) -> MLPParams:

    return {
        "hidden_neurons_size": random.choice([
            parent1["hidden_neurons_size"],
            parent2["hidden_neurons_size"]
        ]),
        "activation": random.choice([
            parent1["activation"],
            parent2["activation"]
        ]),
        "alpha": random.choice([
            parent1["alpha"],
            parent2["alpha"]
        ]),
        "solver": random.choice([
            parent1["solver"],
            parent2["solver"]
        ]),
        "max_iterations": random.choice([
            parent1["max_iterations"],
            parent2["max_iterations"]
        ])
    }


def mutate(
    individual: MLPParams,
    mutation_rate: float,
    hidden_neurons: list[int],
    activations: list[str],
    alphas: list[float],
    solvers: list[str],
    num_iterations: list[int]
) -> MLPParams:

    if random.random() < mutation_rate:
        individual["hidden_neurons_size"] = random.choice(
            hidden_neurons
        )

    if random.random() < mutation_rate:
        individual["activation"] = random.choice(
            activations
        )

    if random.random() < mutation_rate:
        individual["alpha"] = random.choice(
            alphas
        )

    if random.random() < mutation_rate:
        individual["solver"] = random.choice(
            solvers
        )

    if random.random() < mutation_rate:
        individual["max_iterations"] = random.choice(
            num_iterations
        )

    return individual


def get_params(
    generations: int = 10,
    population_size: int = 10,
    mutation_rate: float = 0.2,
    verbose: bool = False
) -> GetParamsReturn:

    hidden_neurons = [3, 5, 10, 15, 20, 25, 30]
    activations = ['relu', 'tanh', 'logistic']
    solvers = ['adam', 'sgd']
    alphas = [0.0001, 0.001, 0.01, 0.1]
    num_iterations = [10, 100, 500, 1000, 2000]

    population = [
        create_individual(
            hidden_neurons,
            activations,
            alphas,
            solvers,
            num_iterations
        )
        for _ in range(population_size)
    ]

    historic: list[HistoricItem] = []

    best_accuracy = 0.0
    best_params = None

    worst_accuracy = 1.1
    worst_params = None

    for generation in range(generations):

        if verbose:
            print()
            print(f"=== Geração {generation + 1} ===")

        evaluated_population: list[
            tuple[MLPParams, float]
        ] = []

        for index, individual in enumerate(population):

            accuracy = evaluate(individual)

            historic.append(
                HistoricItem(
                    params=individual.copy(),
                    accuracy=accuracy
                )
            )

            evaluated_population.append(
                (individual, accuracy)
            )

            if verbose:
                print(
                    f"Indivíduo {index + 1:02d} | "
                    f"Acurácia = {accuracy:.4f}"
                )

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = individual.copy()

            if accuracy < worst_accuracy:
                worst_accuracy = accuracy
                worst_params = individual.copy()

        evaluated_population.sort(
            key=lambda item: item[1],
            reverse=True
        )

        if verbose:
            print(
                f"Melhor da geração: "
                f"{evaluated_population[0][1]:.4f}"
            )

        elite_1 = evaluated_population[0][0]
        elite_2 = evaluated_population[1][0]

        new_population = [
            elite_1.copy(),
            elite_2.copy()
        ]

        while len(new_population) < population_size:

            parent1 = random.choice(
                evaluated_population[:5]
            )[0]

            parent2 = random.choice(
                evaluated_population[:5]
            )[0]

            child = crossover(
                parent1,
                parent2
            )

            child = mutate(
                child,
                mutation_rate,
                hidden_neurons,
                activations,
                alphas,
                solvers,
                num_iterations
            )

            new_population.append(child)

        population = new_population

    if verbose:
        print()
        print(f"Melhor acurácia: {best_accuracy}")
        print(f"Melhores parâmetros: {best_params}")

        print()

        print(f"Pior acurácia: {worst_accuracy}")
        print(f"Piores parâmetros: {worst_params}")

    return {
        "best_accuracy": best_accuracy,
        "best_params": best_params,
        "worst_accuracy": worst_accuracy,
        "worst_params": worst_params,
        "historic": historic
    }


def get_best_mlp(generations: int = 10, population_size: int = 10, mutation_rate: float = 0.2) -> MLPClassifier:
    """
        Obtém a melhor mlp dada a busca genética.
        
        Args:
            generations (int): número de gerações, por padrão é 10
            population_size (int): tamanho da população, por padrão é 10
            mutation_rate (float): taxa de mutação, por padrão é 0.2
        
        Returns:
            MLPClassifier: a melhor rede MLPClassifier
    """
    params: MLPParams = get_params(
        generations=generations,
        population_size=population_size,
        mutation_rate=mutation_rate
    )["best_params"]
    
    mlp: MLPClassifier = MLPClassifier(
        hidden_layer_sizes=params['hidden_neurons_size'],
        activation=params['activation'],
        alpha=params['alpha'],
        solver=params['solver'],
        max_iter=params['max_iterations']
    )
    
    return mlp


def get_worst_mlp(generations: int = 10, population_size: int = 10, mutation_rate: float = 0.2) -> MLPClassifier:
    """
        Obtém a pior mlp dada a busca genética.
        
        Args:
            generations (int): número de gerações, por padrão é 10
            population_size (int): tamanho da população, por padrão é 10
            mutation_rate (float): taxa de mutação, por padrão é 0.2
        
        Returns:
            MLPClassifier: a pior rede MLPClassifier
    """
    params: MLPParams = get_params(
        generations=generations,
        population_size=population_size,
        mutation_rate=mutation_rate
    )["worst_params"]
    
    mlp: MLPClassifier = MLPClassifier(
        hidden_layer_sizes=params['hidden_neurons_size'],
        activation=params['activation'],
        alpha=params['alpha'],
        solver=params['solver'],
        max_iter=params['max_iterations']
    )
    
    return mlp


