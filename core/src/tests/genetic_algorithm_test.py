from mlp import genetic_search

VALORES_VALIDOS = {
    "hidden_neurons_size": [3, 5, 10, 15, 20, 25, 30],
    "activation": ["relu", "tanh", "logistic"],
    "alpha": [0.0001, 0.001, 0.01, 0.1],
    "solver": ["adam", "sgd"],
    "max_iterations": [10, 100, 500, 1000, 2000],
}

GERACOES_RAPIDAS = 2
POPULACAO_RAPIDA = 4

def test_create_individual_retorna_dict():
    individuo = genetic_search.create_individual(
        VALORES_VALIDOS["hidden_neurons_size"],
        VALORES_VALIDOS["activation"],
        VALORES_VALIDOS["alpha"],
        VALORES_VALIDOS["solver"],
        VALORES_VALIDOS["max_iterations"]
    )

    assert isinstance(individuo, dict)
    
def test_create_individual_possui_todas_as_chaves():
    individuo = genetic_search.create_individual(
        VALORES_VALIDOS["hidden_neurons_size"],
        VALORES_VALIDOS["activation"],
        VALORES_VALIDOS["alpha"],
        VALORES_VALIDOS["solver"],
        VALORES_VALIDOS["max_iterations"]
    )

    assert set(individuo.keys()) == set(VALORES_VALIDOS.keys())
    

def test_create_individual_valores_validos():
    individuo = genetic_search.create_individual(
        VALORES_VALIDOS["hidden_neurons_size"],
        VALORES_VALIDOS["activation"],
        VALORES_VALIDOS["alpha"],
        VALORES_VALIDOS["solver"],
        VALORES_VALIDOS["max_iterations"]
    )

    for chave, validos in VALORES_VALIDOS.items():
        assert individuo[chave] in validos
        

def test_evaluate_retorna_float():
    individuo = genetic_search.create_individual(
        VALORES_VALIDOS["hidden_neurons_size"],
        VALORES_VALIDOS["activation"],
        VALORES_VALIDOS["alpha"],
        VALORES_VALIDOS["solver"],
        VALORES_VALIDOS["max_iterations"]
    )

    resultado = genetic_search.evaluate(individuo)

    assert isinstance(resultado, float)
    
def test_evaluate_retorna_valor_entre_0_e_1():
    individuo = genetic_search.create_individual(
        VALORES_VALIDOS["hidden_neurons_size"],
        VALORES_VALIDOS["activation"],
        VALORES_VALIDOS["alpha"],
        VALORES_VALIDOS["solver"],
        VALORES_VALIDOS["max_iterations"]
    )

    accuracy = genetic_search.evaluate(individuo)

    assert 0.0 <= accuracy <= 1.0
    
def test_crossover_retorna_dict():
    pai1 = {
        "hidden_neurons_size": 5,
        "activation": "relu",
        "alpha": 0.001,
        "solver": "adam",
        "max_iterations": 100
    }

    pai2 = {
        "hidden_neurons_size": 20,
        "activation": "tanh",
        "alpha": 0.1,
        "solver": "sgd",
        "max_iterations": 1000
    }

    filho = genetic_search.crossover(pai1, pai2)

    assert isinstance(filho, dict)
    
def test_crossover_herda_apenas_genes_dos_pais():
    pai1 = {
        "hidden_neurons_size": 5,
        "activation": "relu",
        "alpha": 0.001,
        "solver": "adam",
        "max_iterations": 100
    }

    pai2 = {
        "hidden_neurons_size": 20,
        "activation": "tanh",
        "alpha": 0.1,
        "solver": "sgd",
        "max_iterations": 1000
    }

    filho = genetic_search.crossover(pai1, pai2)

    for chave in filho:
        assert filho[chave] in [pai1[chave], pai2[chave]]


def test_mutate_com_taxa_zero_nao_altera():
    individuo = {
        "hidden_neurons_size": 5,
        "activation": "relu",
        "alpha": 0.001,
        "solver": "adam",
        "max_iterations": 100
    }

    original = individuo.copy()

    resultado = genetic_search.mutate(
        individuo,
        0.0,
        VALORES_VALIDOS["hidden_neurons_size"],
        VALORES_VALIDOS["activation"],
        VALORES_VALIDOS["alpha"],
        VALORES_VALIDOS["solver"],
        VALORES_VALIDOS["max_iterations"]
    )

    assert resultado == original
    

def test_mutate_com_taxa_um_mantem_valores_validos():
    individuo = {
        "hidden_neurons_size": 5,
        "activation": "relu",
        "alpha": 0.001,
        "solver": "adam",
        "max_iterations": 100
    }

    resultado = genetic_search.mutate(
        individuo,
        1.0,
        VALORES_VALIDOS["hidden_neurons_size"],
        VALORES_VALIDOS["activation"],
        VALORES_VALIDOS["alpha"],
        VALORES_VALIDOS["solver"],
        VALORES_VALIDOS["max_iterations"]
    )

    for chave, validos in VALORES_VALIDOS.items():
        assert resultado[chave] in validos
        

def test_get_params_retorna_dict():
    resultado = genetic_search.get_params(
        generations=GERACOES_RAPIDAS,
        population_size=POPULACAO_RAPIDA
    )

    assert isinstance(resultado, dict)
    

def test_get_params_possui_todas_as_chaves():
    resultado = genetic_search.get_params(
        generations=GERACOES_RAPIDAS,
        population_size=POPULACAO_RAPIDA
    )

    esperado = {
        "best_accuracy",
        "best_params",
        "worst_accuracy",
        "worst_params",
        "historic"
    }

    assert set(resultado.keys()) == esperado
    

def test_historico_e_lista():
    resultado = genetic_search.get_params(
        generations=GERACOES_RAPIDAS,
        population_size=POPULACAO_RAPIDA
    )

    assert isinstance(resultado["historic"], list)
    

def test_historico_tem_tamanho_correto():
    geracoes = 3
    populacao = 4

    resultado = genetic_search.get_params(
        generations=geracoes,
        population_size=populacao
    )

    esperado = geracoes * populacao

    assert len(resultado["historic"]) == esperado
    

def test_best_accuracy_maior_ou_igual_worst_accuracy():
    resultado = genetic_search.get_params(
        generations=GERACOES_RAPIDAS,
        population_size=POPULACAO_RAPIDA
    )

    assert (
        resultado["best_accuracy"]
        >=
        resultado["worst_accuracy"]
    )


def test_best_accuracy_bate_com_historico():
    resultado = genetic_search.get_params(
        generations=GERACOES_RAPIDAS,
        population_size=POPULACAO_RAPIDA
    )

    maior = max(
        item.accuracy
        for item in resultado["historic"]
    )

    assert resultado["best_accuracy"] == maior
    

def test_worst_accuracy_bate_com_historico():
    resultado = genetic_search.get_params(
        generations=GERACOES_RAPIDAS,
        population_size=POPULACAO_RAPIDA
    )

    menor = min(
        item.accuracy
        for item in resultado["historic"]
    )

    assert resultado["worst_accuracy"] == menor
    

def test_cada_item_do_historico_tem_params_e_accuracy():
    resultado = genetic_search.get_params(
        generations=GERACOES_RAPIDAS,
        population_size=POPULACAO_RAPIDA
    )

    for item in resultado["historic"]:
        assert hasattr(item, "params")
        assert hasattr(item, "accuracy")


def test_get_params_best_e_worst_params_nao_sao_none():
    resultado = genetic_search.get_params(
        generations=GERACOES_RAPIDAS,
        population_size=POPULACAO_RAPIDA
    )

    assert resultado["best_params"] is not None
    assert resultado["worst_params"] is not None


def test_get_params_best_e_worst_params_tem_valores_validos():
    resultado = genetic_search.get_params(
        generations=GERACOES_RAPIDAS,
        population_size=POPULACAO_RAPIDA
    )

    for params in [resultado["best_params"], resultado["worst_params"]]:
        for chave, validos in VALORES_VALIDOS.items():
            assert params[chave] in validos


def test_get_params_verbose_true_nao_quebra(capsys):
    genetic_search.get_params(
        generations=1,
        population_size=4,
        verbose=True
    )

    saida = capsys.readouterr().out

    assert "Geração" in saida
    assert "Melhor acurácia" in saida


def test_get_params_com_populacao_minima_dois():
    resultado = genetic_search.get_params(
        generations=1,
        population_size=2
    )

    assert len(resultado["historic"]) == 2
    assert 0.0 <= resultado["best_accuracy"] <= 1.0
