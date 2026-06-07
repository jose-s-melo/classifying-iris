"""
Testes para o módulo random_search.py
Cada função de teste verifica uma etapa por vez.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'mpl')))
import random_search 

from sklearn.neural_network import MLPClassifier

VALORES_VALIDOS = {
    "hidden_neurons_size": [3, 5, 10, 15, 20, 25, 30],
    "activation":          ["relu", "tanh", "logistic"],
    "alpha":               [0.0001, 0.001, 0.01, 0.1],
    "solver":              ["adam", "sgd"],
    "max_iterations":      [10, 100, 500, 1000, 2000],
}

NUM_BUSCAS_RAPIDO = 5   # valor pequeno = agilidade.

def test_get_params_retorna_dicionario():
    """get_params() deve retornar um dicionário (dict)."""
    resultado = random_search.get_params(num_searchs=NUM_BUSCAS_RAPIDO)

    assert isinstance(resultado, dict), (
        f"Esperava dict, mas recebeu {type(resultado)}"
    )

    print("OK  test_get_params_retorna_dicionario")


def test_get_params_possui_todas_as_chaves():
    """O dicionário retornado deve ter exatamente as 5 chaves esperadas."""
    resultado = random_search.get_params(num_searchs=NUM_BUSCAS_RAPIDO)

    chaves_esperadas = {"best_accuracy", "best_params", "worst_accuracy", "worst_params", "historic"}
    chaves_recebidas = set(resultado.keys())

    assert chaves_recebidas == chaves_esperadas, (
        f"Chaves erradas.\n  Esperado: {chaves_esperadas}\n  Recebido: {chaves_recebidas}"
    )

    print("OK  test_get_params_possui_todas_as_chaves")


def test_get_params_historic_e_lista():
    """O campo 'historic' deve ser uma lista."""
    resultado = random_search.get_params(num_searchs=NUM_BUSCAS_RAPIDO)

    assert isinstance(resultado["historic"], list), (
        f"'historic' deveria ser list, mas é {type(resultado['historic'])}"
    )

    print("OK  test_get_params_historic_e_lista")


def test_get_params_historic_tem_tamanho_correto():
    """
    O histórico deve ter exatamente `num_searchs` itens,
    pois cada busca gera uma entrada no histórico.
    """
    resultado = random_search.get_params(num_searchs=NUM_BUSCAS_RAPIDO)

    tamanho = len(resultado["historic"])

    assert tamanho == NUM_BUSCAS_RAPIDO, (
        f"Esperava {NUM_BUSCAS_RAPIDO} itens no histórico, mas há {tamanho}"
    )

    print("OK  test_get_params_historic_tem_tamanho_correto")

def test_best_params_tem_todas_as_chaves():
    """
    Os melhores parâmetros devem conter todas as chaves que a MLP precisa:
    hidden_neurons_size, activation, alpha, solver, max_iterations.
    """
    resultado = random_search.get_params(num_searchs=NUM_BUSCAS_RAPIDO)
    best_params = resultado["best_params"]

    chaves_esperadas = set(VALORES_VALIDOS.keys())
    chaves_recebidas = set(best_params.keys())

    assert chaves_recebidas == chaves_esperadas, (
        f"best_params com chaves erradas.\n  Esperado: {chaves_esperadas}\n  Recebido: {chaves_recebidas}"
    )

    print("OK  test_best_params_tem_todas_as_chaves")


def test_worst_params_tem_todas_as_chaves():
    """Mesma verificação de chaves, mas para os piores parâmetros."""
    resultado = random_search.get_params(num_searchs=NUM_BUSCAS_RAPIDO)
    worst_params = resultado["worst_params"]

    chaves_esperadas = set(VALORES_VALIDOS.keys())
    chaves_recebidas = set(worst_params.keys())

    assert chaves_recebidas == chaves_esperadas, (
        f"worst_params com chaves erradas.\n  Esperado: {chaves_esperadas}\n  Recebido: {chaves_recebidas}"
    )

    print("OK  test_worst_params_tem_todas_as_chaves")


def test_best_params_valores_dentro_do_espaco_de_busca():
    """
    Cada valor de best_params deve pertencer à lista de valores válidos.
    Isso garante que o algoritmo não inventou valores fora do espaço definido.
    """
    resultado = random_search.get_params(num_searchs=NUM_BUSCAS_RAPIDO)
    best_params = resultado["best_params"]

    for chave, valores_validos in VALORES_VALIDOS.items():
        valor = best_params[chave]
        assert valor in valores_validos, (
            f"best_params['{chave}'] = {valor!r} não está entre os valores válidos: {valores_validos}"
        )

    print("OK  test_best_params_valores_dentro_do_espaco_de_busca")


def test_worst_params_valores_dentro_do_espaco_de_busca():
    """Mesma verificação de valores válidos para os piores parâmetros."""
    resultado = random_search.get_params(num_searchs=NUM_BUSCAS_RAPIDO)
    worst_params = resultado["worst_params"]

    for chave, valores_validos in VALORES_VALIDOS.items():
        valor = worst_params[chave]
        assert valor in valores_validos, (
            f"worst_params['{chave}'] = {valor!r} não está entre os valores válidos: {valores_validos}"
        )

    print("OK  test_worst_params_valores_dentro_do_espaco_de_busca")


def test_best_accuracy_e_float_entre_0_e_1():
    """
    A acurácia é uma proporção: deve ser float no intervalo [0.0, 1.0].
    Valores fora disso indicam erro no cálculo.
    """
    resultado = random_search.get_params(num_searchs=NUM_BUSCAS_RAPIDO)
    best_accuracy = resultado["best_accuracy"]

    assert isinstance(best_accuracy, float), (
        f"best_accuracy deveria ser float, mas é {type(best_accuracy)}"
    )
    assert 0.0 <= best_accuracy <= 1.0, (
        f"best_accuracy fora do intervalo [0, 1]: {best_accuracy}"
    )

    print("OK  test_best_accuracy_e_float_entre_0_e_1")


def test_worst_accuracy_e_float_entre_0_e_1():
    """Mesma verificação de intervalo para a pior acurácia."""
    resultado = random_search.get_params(num_searchs=NUM_BUSCAS_RAPIDO)
    worst_accuracy = resultado["worst_accuracy"]

    assert isinstance(worst_accuracy, float), (
        f"worst_accuracy deveria ser float, mas é {type(worst_accuracy)}"
    )
    assert 0.0 <= worst_accuracy <= 1.0, (
        f"worst_accuracy fora do intervalo [0, 1]: {worst_accuracy}"
    )

    print("OK  test_worst_accuracy_e_float_entre_0_e_1")


def test_best_accuracy_maior_ou_igual_a_worst_accuracy():
    """
    Por definição, a melhor acurácia nunca pode ser menor que a pior.
    Se isso acontecer, a lógica de comparação dentro de get_params() está errada.
    """
    resultado = random_search.get_params(num_searchs=NUM_BUSCAS_RAPIDO)

    best  = resultado["best_accuracy"]
    worst = resultado["worst_accuracy"]

    assert best >= worst, (
        f"best_accuracy ({best}) deveria ser >= worst_accuracy ({worst})"
    )

    print("OK  test_best_accuracy_maior_ou_igual_a_worst_accuracy")

def test_best_accuracy_bate_com_maximo_do_historico():
    """
    A best_accuracy declarada deve ser igual ao maior valor encontrado
    percorrendo o histórico manualmente. Valida a lógica de rastreamento.
    """
    resultado = random_search.get_params(num_searchs=NUM_BUSCAS_RAPIDO)

    maior_no_historico = max(item.accuracy for item in resultado["historic"])

    assert resultado["best_accuracy"] == maior_no_historico, (
        f"best_accuracy ({resultado['best_accuracy']}) difere do máximo "
        f"calculado no histórico ({maior_no_historico})"
    )

    print("OK  test_best_accuracy_bate_com_maximo_do_historico")


def test_worst_accuracy_bate_com_minimo_do_historico():
    """
    A worst_accuracy declarada deve ser igual ao menor valor encontrado
    percorrendo o histórico manualmente.
    """
    resultado = random_search.get_params(num_searchs=NUM_BUSCAS_RAPIDO)

    menor_no_historico = min(item.accuracy for item in resultado["historic"])

    assert resultado["worst_accuracy"] == menor_no_historico, (
        f"worst_accuracy ({resultado['worst_accuracy']}) difere do mínimo "
        f"calculado no histórico ({menor_no_historico})"
    )

    print("OK  test_worst_accuracy_bate_com_minimo_do_historico")


def test_cada_item_do_historico_tem_accuracy_e_params():
    """
    Cada HistoricItem deve ter os atributos 'accuracy' e 'params'.
    Um item sem esses atributos quebraria o main_test_1().
    """
    resultado = random_search.get_params(num_searchs=NUM_BUSCAS_RAPIDO)

    for i, item in enumerate(resultado["historic"]):
        assert hasattr(item, "accuracy"), f"Item {i} do histórico sem atributo 'accuracy'"
        assert hasattr(item, "params"),   f"Item {i} do histórico sem atributo 'params'"

    print("OK  test_cada_item_do_historico_tem_accuracy_e_params")

def test_get_best_mlp_retorna_mlpclassifier():
    """get_best_mlp() deve retornar um objeto MLPClassifier do sklearn."""
    mlp = random_search.get_best_mlp(num_searchs=NUM_BUSCAS_RAPIDO)

    assert isinstance(mlp, MLPClassifier), (
        f"get_best_mlp() deveria retornar MLPClassifier, mas retornou {type(mlp)}"
    )

    print("OK  test_get_best_mlp_retorna_mlpclassifier")


def test_get_worst_mlp_retorna_mlpclassifier():
    """get_worst_mlp() deve retornar um objeto MLPClassifier do sklearn."""
    mlp = random_search.get_worst_mlp(num_searchs=NUM_BUSCAS_RAPIDO)

    assert isinstance(mlp, MLPClassifier), (
        f"get_worst_mlp() deveria retornar MLPClassifier, mas retornou {type(mlp)}"
    )

    print("OK  test_get_worst_mlp_retorna_mlpclassifier")


def test_get_best_mlp_usa_parametros_validos():
    """
    Os hiperparâmetros configurados na MLP retornada por get_best_mlp()
    devem ser valores pertencentes ao espaço de busca definido.
    """
    mlp = random_search.get_best_mlp(num_searchs=NUM_BUSCAS_RAPIDO)

    neuronio = mlp.hidden_layer_sizes if isinstance(mlp.hidden_layer_sizes, int) else mlp.hidden_layer_sizes[0]

    assert neuronio in VALORES_VALIDOS["hidden_neurons_size"], (
        f"hidden_layer_sizes inválido: {neuronio}"
    )
    assert mlp.activation in VALORES_VALIDOS["activation"], (
        f"activation inválido: {mlp.activation}"
    )
    assert mlp.alpha in VALORES_VALIDOS["alpha"], (
        f"alpha inválido: {mlp.alpha}"
    )
    assert mlp.solver in VALORES_VALIDOS["solver"], (
        f"solver inválido: {mlp.solver}"
    )
    assert mlp.max_iter in VALORES_VALIDOS["max_iterations"], (
        f"max_iter inválido: {mlp.max_iter}"
    )

    print("OK  test_get_best_mlp_usa_parametros_validos")

def test_get_params_com_uma_busca():
    """
    Com apenas 1 busca, best e worst devem apontar para o mesmo resultado.
    Testa o caso limite mínimo da função.
    """
    resultado = random_search.get_params(num_searchs=1)

    assert len(resultado["historic"]) == 1, (
        "Histórico deveria ter exatamente 1 item"
    )
    assert resultado["best_accuracy"] == resultado["worst_accuracy"], (
        "Com 1 busca, best e worst deveriam ser iguais"
    )

    print("OK  test_get_params_com_uma_busca")

TODOS_OS_TESTES = [
    test_get_params_retorna_dicionario,
    test_get_params_possui_todas_as_chaves,
    test_get_params_historic_e_lista,
    test_get_params_historic_tem_tamanho_correto,

    test_best_params_tem_todas_as_chaves,
    test_worst_params_tem_todas_as_chaves,
    test_best_params_valores_dentro_do_espaco_de_busca,
    test_worst_params_valores_dentro_do_espaco_de_busca,

    test_best_accuracy_e_float_entre_0_e_1,
    test_worst_accuracy_e_float_entre_0_e_1,
    test_best_accuracy_maior_ou_igual_a_worst_accuracy,

    test_best_accuracy_bate_com_maximo_do_historico,
    test_worst_accuracy_bate_com_minimo_do_historico,
    test_cada_item_do_historico_tem_accuracy_e_params,

    test_get_best_mlp_retorna_mlpclassifier,
    test_get_worst_mlp_retorna_mlpclassifier,
    test_get_best_mlp_usa_parametros_validos,

    test_get_params_com_uma_busca,
]


if __name__ == "__main__":
    passou = 0
    falhou = 0
    falhas = []

    print("=" * 60)
    print("  Executando testes — MLP Random Search")
    print("=" * 60)

    for teste in TODOS_OS_TESTES:
        try:
            teste()
            passou += 1
        except AssertionError as erro:
            falhou += 1
            falhas.append((teste.__name__, str(erro)))
            print(f"FALHOU  {teste.__name__}")

    print()
    print("=" * 60)
    print(f"  Resultado: {passou} APTOS | {falhou} INAPTOS")
    print("=" * 60)

    if falhas:
        print("\nDetalhes das falhas:")
        for nome, mensagem in falhas:
            print(f"\n  [{nome}]\n  {mensagem}")