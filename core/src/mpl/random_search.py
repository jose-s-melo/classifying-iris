import random
from sklearn.neural_network import MLPClassifier


from sklearn.datasets import load_iris
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import random


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


hidden_neurons = [3, 5, 10, 15, 20, 25, 30]
activations = ['relu', 'tanh', 'logistic']
solvers = ['adam', 'sgd']
alphas = [0.0001, 0.001, 0.01, 0.1]
num_iterations = [10, 100, 500, 1000, 2000]

TESTS = 50

best_accuracy = 0
best_params = None

worst_accuracy = 1.1
worst_params = None

for i in range(TESTS):
    
    params = {
        "hidden_neurons_size": random.choice(hidden_neurons),
        "activation": random.choice(activations),
        "alpha": random.choice(alphas),
        "solver": random.choice(solvers),
        "max_iterations": random.choice(num_iterations)
    }
    
    mlp = MLPClassifier(
        hidden_layer_sizes=(params["hidden_neurons_size"],),
        activation=params["activation"],
        solver=params["solver"],
        alpha=params["alpha"],
        max_iter=params["max_iterations"]
    )
    
    mlp.fit(X_train, Y_train)

    predictions = mlp.predict(X_test)

    accuracy = accuracy_score(Y_test, predictions)

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
        

print()
print(f"Melhor acurácia: {best_accuracy}")
print(f"Melhores parâmetros: {best_params}")
print()
print(f"Pior acurácia: {worst_accuracy}")
print(f"Piores parâmetros: {worst_params}")