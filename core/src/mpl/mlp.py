from sklearn.   datasets import load_iris
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


dataset = load_iris()

"""
O scaler é o 
"""
scaler = RobustScaler()

X = dataset['data']
Y = dataset['target']
target_names = dataset['target_names']
feature_names = dataset['feature_names']


X_train, X_test, Y_train, Y_test = train_test_split(
    X, 
    Y, 
    test_size=0.3)


X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)


for i in range(10, 2000, 50):
    mlp = MLPClassifier(
        hidden_layer_sizes=(10,),
        activation='relu',
        solver='sgd',
        alpha=0.0001,
        max_iter=i
    )
    mlp.fit(X=X_train, y=Y_train)

    Y_predictions = mlp.predict(X_test)

    accuracy = accuracy_score(Y_test, Y_predictions)


    print(accuracy)
    


