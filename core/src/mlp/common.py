from typing import TypedDict, Literal
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

MLPScaler = StandardScaler | MinMaxScaler | RobustScaler

class MLPParams(TypedDict):
    hidden_neurons_size: int
    activation: Literal['relu', 'tanh', 'logistic']
    alpha: float
    solver: Literal['adam', 'sgd']
    max_iterations: int
    scaler: Literal['standard', 'minmax', 'robust']
    

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