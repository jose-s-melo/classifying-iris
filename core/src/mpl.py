from sklearn.datasets import load_iris

dataset = load_iris()

# print(dataset)

iris_data = dataset.data

iris_target = dataset.target

iris_labels = dataset.target_names

print(dataset.DESCR)