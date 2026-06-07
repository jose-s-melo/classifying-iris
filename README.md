# Classifying Iris

Rede neural MLP com uma camada para classificar os três tipos de flores do Iris Dataset. Projeto desenvolvido como parte das atividades da disciplina Inteligência Artificial @ UFCG.


## Descrição do trabalho

Treinar uma rede neural artificial MLP com apenas 1 camada escondida para classificar as 3 categorias de flores da base de dados iris-dataset. Utilize um algoritmo genético simples para fazer a busca pelos melhores parâmetros de treinamento da rede. Faça outra sequência de treinamentos fazendo a busca dos parâmetros escolhendo valores aleatoriamente a partir de uma faixa predefinida de valores

---

# Executar

Para executar use:
```bash
cd core/src/
uv run python -m arquivo_para_executar

# Exemplo

uv run python -m mlp.random_search
```

---

Para executar os testes:
```bash
cd core/src/
uv run python -m pytest
```

Para verificar a cobertura:
```bash
cd core/src/
uv run python -m pytest
coverage run -m pytest
coverage report -m
```