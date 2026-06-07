import random_search
import genetic_search

def main_test_1():
    params = random_search.get_params()
    
    for elem in params['historic']:
        print(elem.accuracy)
        print(elem.params)
        print()

def main_test_2():
    worst_mlp = random_search.get_worst_mlp(10)
    
    best_mlp = random_search.get_best_mlp(10)

def main_test_3():
    params = genetic_search.get_params(
        generations=5,
        population_size=10
    )

    for elem in params['historic']:
        print(elem.accuracy)
        print(elem.params)
        print()

def main_test_4():
    worst_mlp = genetic_search.get_worst_mlp(
        generations=5,
        population_size=10
    )

    best_mlp = genetic_search.get_best_mlp(
        generations=5,
        population_size=10
    )

    print(best_mlp)
    print()
    print(worst_mlp)

def main_test_5():
    params = genetic_search.get_params(
        generations=5,
        population_size=10,
        verbose=True
    )

if __name__ == '__main__':
    main_test_3()