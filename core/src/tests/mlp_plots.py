from typing import Dict, List, Any
from collections import defaultdict
import matplotlib.pyplot as plot
from mlp.random_search import get_params, GetParamsReturn, HistoricItem
from pathlib import Path


PATH_TO_SAVE = Path("./plots")
PATH_TO_SAVE.mkdir(exist_ok=True)

def calculate_mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def process_and_plot_results():
    PARAM_CONFIG = {
        "activation": ("Activation Functions", "activation"),
        "alpha": ("Alpha Values", "alpha"),
        "solver": ("Solvers", "solver"),
        "hidden_neurons_size": ("Hidden Neuron Size", "hidden_neuron_size"),
        "max_iterations": ("Max Iterations", "max_iterations"),
    }

    random_search_results: GetParamsReturn = get_params(num_searchs=100)
    param_results: List[HistoricItem] = random_search_results["historic"]

    grouped_accuracies: Dict[str, Dict[Any, List[float]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for result in param_results:
        params = (
            result.params if isinstance(result.params, dict) else vars(result.params)
        )

        for param_key in PARAM_CONFIG.keys():
            param_value = params.get(param_key)
            if param_value is not None:
                grouped_accuracies[param_key][str(param_value)].append(result.accuracy)

    for param_key, (title, file_prefix) in PARAM_CONFIG.items():
        param_data = grouped_accuracies.get(param_key)
        if not param_data:
            continue

        x_values = list(param_data.keys())
        y_values = [calculate_mean(acc_list) for acc_list in param_data.values()]

        plot.clf()
        plot.bar(x_values, y_values)
        plot.title(f"Mean Accuracy by {title}")
        plot.xlabel(title)
        plot.ylabel("Accuracy")
        plot.savefig(PATH_TO_SAVE / f"{file_prefix}_bar.png")

        plot.clf()
        for param_value, accuracies in param_data.items():
            plot.hist(
                accuracies, alpha=0.5, label=f"{param_key}: {param_value}", bins=10
            )

        plot.title(f"Accuracy Distribution by {title}")
        plot.xlabel("Accuracy")
        plot.ylabel("Frequency")
        plot.legend()
        plot.savefig(PATH_TO_SAVE / f"{file_prefix}_hist.png")


if __name__ == "__main__":
    process_and_plot_results()