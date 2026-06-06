import random_search

def main_test_1():
    params = random_search.get_params()
    
    for elem in params['historic']:
        print(elem.accuracy)
        print(elem.params)
        print()

def main_test_2():
    worst_mlp = random_search.get_worst_mlp(10)
    
    best_mlp = random_search.get_best_mlp(10)
    

if __name__ == '__main__':
    main_test_2()