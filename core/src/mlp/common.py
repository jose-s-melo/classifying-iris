from typing import TypedDict, Literal

class MLPParams(TypedDict):
    hidden_neurons_size: int
    activation: Literal['relu', 'tanh', 'logistic']
    alpha: float
    solver: Literal['adam', 'sgd']
    max_iterations: int
    

class HistoricItem:
    def __init__(self, params: MLPParams, accuracy: float):
        self.params = params
        self.accuracy = accuracy
        
    params: MLPParams
    accuracy: float

class GetParamsReturn(TypedDict):
    best_accuracy: float
    best_params: MLPParams
    worst_accuracy: float
    worst_params: MLPParams
    historic: list[HistoricItem]