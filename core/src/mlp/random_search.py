import random
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from mlp.common import MLPParams, GetParamsReturn, HistoricItem, MLPScaler

import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings(
    "ignore",
    category=ConvergenceWarning
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


def get_params(num_searchs: int = 50, verbose: bool = False) -> GetParamsReturn:
    """
        Obtém os parâmetros a serem usados na MLP,
        a função retorna tanto os melhores parâmetros quanto
        os piores.
        
        Args:
            num_searchs (int): número de buscas a serem realizados, por padrão é 50
            verbose (bool): permite mostrar o processo do algoritmo, por padão é falso

        Returns:
            GetParamsReturn: dicionário com as chaves: best_param, best_accuracy, worst_params, worst_accuracy e o histórico.
    """
    
    hidden_neurons = [3, 5, 10, 15, 20, 25, 30]
    activations = ['relu', 'tanh', 'logistic']
    solvers = ['adam', 'sgd']
    alphas = [0.0001, 0.001, 0.01, 0.1]
    num_iterations = [10, 100, 500, 1000, 2000]
    scalers = ['standard', 'minmax', 'robust']

    best_accuracy = 0
    best_params = None

    worst_accuracy = 1.1
    worst_params = None
    
    historic: list[HistoricItem] = []

    for i in range(num_searchs):
        params: MLPParams = {
            "hidden_neurons_size": random.choice(hidden_neurons),
            "activation": random.choice(activations),
            "alpha": random.choice(alphas),
            "solver": random.choice(solvers),
            "max_iterations": random.choice(num_iterations),
            "scaler": random.choice(scalers)
        }
        
        scaler_name = params["scaler"]
        if scaler_name == "standard":
            scaler = StandardScaler()
        elif scaler_name == "minmax":
            scaler = MinMaxScaler()
        else:
            scaler = RobustScaler()
            
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        mlp = MLPClassifier(
            hidden_layer_sizes=(params["hidden_neurons_size"],),
            activation=params["activation"],
            solver=params["solver"],
            alpha=params["alpha"],
            max_iter=params["max_iterations"]
        )
        
        mlp.fit(X_train_scaled, Y_train)

        predictions = mlp.predict(X_test_scaled)

        accuracy = accuracy_score(Y_test, predictions)
        
        historic.append(HistoricItem(params, accuracy))

        if verbose:
            print(
                f"Teste {i+1:02d} | "
                f"Acurácia = {accuracy:.4f}"
            )

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = params
            
        if accuracy < worst_accuracy:
            worst_accuracy = accuracy
            worst_params = params
            

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


def get_best_mlp(num_searchs: int) -> tuple[MLPClassifier, MLPScaler]:
    params: MLPParams = get_params(num_searchs)["best_params"]
    
    mlp = MLPClassifier(
        hidden_layer_sizes=params['hidden_neurons_size'],
        activation=params['activation'],
        alpha=params['alpha'],
        solver=params['solver'],
        max_iter=params['max_iterations']
    )
    
    scaler_name = params["scaler"]
    if scaler_name == "standard":
        scaler = StandardScaler()
    elif scaler_name == "minmax":
        scaler = MinMaxScaler()
    else:
        scaler = RobustScaler()
        
    return mlp, scaler


def get_worst_mlp(num_searchs: int) -> tuple[MLPClassifier, MLPScaler]:
    params: MLPParams = get_params(num_searchs)["worst_params"]
    
    mlp = MLPClassifier(
        hidden_layer_sizes=params['hidden_neurons_size'],
        activation=params['activation'],
        alpha=params['alpha'],
        solver=params['solver'],
        max_iter=params['max_iterations']
    )
    
    scaler_name = params["scaler"]
    if scaler_name == "standard":
        scaler = StandardScaler()
    elif scaler_name == "minmax":
        scaler = MinMaxScaler()
    else:
        scaler = RobustScaler()
        
    return mlp, scaler
    
