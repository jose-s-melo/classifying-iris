from typing import Dict, List
from collections import defaultdict
import matplotlib.pyplot as plot
import numpy as np
import warnings
from sklearn.exceptions import ConvergenceWarning
from mlp.random_search import get_params as get_random_params
from mlp.genetic_search import get_params as get_genetic_params
from mlp.common import GetParamsReturn, HistoricItem
from pathlib import Path

warnings.filterwarnings("ignore", category=ConvergenceWarning)

PATH_TO_SAVE = Path(__file__).resolve().parent.parent / "plots"
PATH_TO_SAVE.mkdir(exist_ok=True)


def calculate_mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def calculate_median(values: List[float]) -> float:
    return float(np.median(values)) if values else 0.0


def get_running_max(historic: List[HistoricItem]) -> List[float]:
    running_max = []
    current_max = 0.0
    for item in historic:
        current_max = max(current_max, item.accuracy)
        running_max.append(current_max)
    return running_max


def get_generational_best(
    historic: List[HistoricItem], population_size: int
) -> List[float]:
    generational_best = []
    current_max = 0.0
    if not historic or population_size <= 0:
        return []
    num_generations = len(historic) // population_size
    for g in range(num_generations):
        generation_slice = historic[g * population_size : (g + 1) * population_size]
        if generation_slice:
            generation_max = max(item.accuracy for item in generation_slice)
            current_max = max(current_max, generation_max)
        generational_best.append(current_max)
    return generational_best


def group_accuracies_by_param(
    historic: List[HistoricItem], param_keys: List[str]
) -> Dict[str, Dict[str, List[float]]]:
    grouped = defaultdict(lambda: defaultdict(list))
    for result in historic:
        params = (
            result.params if isinstance(result.params, dict) else vars(result.params)
        )
        for param_key in param_keys:
            param_value = params.get(param_key)
            if param_value is not None:
                grouped[param_key][str(param_value)].append(result.accuracy)
    return grouped


def generate_median_search_comparison_plot(num_samples: int = 15):
    from concurrent.futures import ProcessPoolExecutor

    print(
        f"Running {num_samples} samples of Random Search and Genetic Search in parallel..."
    )
    with ProcessPoolExecutor() as executor:
        random_futures = [
            executor.submit(get_random_params, 15) for _ in range(num_samples)
        ]
        genetic_futures = [
            executor.submit(get_genetic_params, 3, 5) for _ in range(num_samples)
        ]

        random_results = [f.result() for f in random_futures]
        genetic_results = [f.result() for f in genetic_futures]

    random_runs_running_max = []
    for res in random_results:
        running_max = get_running_max(res["historic"])
        random_runs_running_max.append(running_max)

    genetic_runs_running_max = []
    for res in genetic_results:
        running_max = get_running_max(res["historic"])
        genetic_runs_running_max.append(running_max)

    random_runs_running_max = np.array(random_runs_running_max)
    random_medians = np.median(random_runs_running_max, axis=0)
    random_means = np.mean(random_runs_running_max, axis=0)

    genetic_runs_running_max = np.array(genetic_runs_running_max)
    genetic_medians = np.median(genetic_runs_running_max, axis=0)
    genetic_means = np.mean(genetic_runs_running_max, axis=0)

    plot.figure(figsize=(10, 6))
    plot.grid(True, linestyle="--", alpha=0.5)

    steps_random = range(1, len(random_medians) + 1)
    steps_genetic = range(1, len(genetic_medians) + 1)

    color_random = "#E63946"
    color_genetic = "#1D3557"

    plot.plot(
        steps_random,
        random_medians,
        label="Random Search (Median)",
        color=color_random,
        linewidth=2.5,
    )
    plot.plot(
        steps_random,
        random_means,
        label="Random Search (Mean)",
        color=color_random,
        linestyle=":",
        linewidth=2.0,
        alpha=0.85,
    )

    random_q25 = np.percentile(random_runs_running_max, 25, axis=0)
    random_q75 = np.percentile(random_runs_running_max, 75, axis=0)
    plot.fill_between(
        steps_random,
        random_q25,
        random_q75,
        color=color_random,
        alpha=0.15,
        label="Random Search (IQR)",
    )

    plot.plot(
        steps_genetic,
        genetic_medians,
        label="Genetic Search (Median)",
        color=color_genetic,
        linewidth=2.5,
    )
    plot.plot(
        steps_genetic,
        genetic_means,
        label="Genetic Search (Mean)",
        color=color_genetic,
        linestyle=":",
        linewidth=2.0,
        alpha=0.85,
    )

    genetic_q25 = np.percentile(genetic_runs_running_max, 25, axis=0)
    genetic_q75 = np.percentile(genetic_runs_running_max, 75, axis=0)
    plot.fill_between(
        steps_genetic,
        genetic_q25,
        genetic_q75,
        color=color_genetic,
        alpha=0.15,
        label="Genetic Search (IQR)",
    )

    plot.title(
        f"Search Algorithm Performance Comparison ({num_samples} runs)",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    plot.xlabel(
        "Cumulative Evaluations (Model Fits)",
        fontsize=12,
        labelpad=10,
    )
    plot.ylabel("Accuracy", fontsize=12, labelpad=10)
    plot.ylim(bottom=0.0, top=1.05)
    plot.xticks(np.arange(0, 16, 2))

    ax = plot.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")

    plot.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#e0e0e0")
    plot.tight_layout()
    plot.savefig(PATH_TO_SAVE / "median_search_comparison.png", dpi=300)
    plot.close()
    print(
        f"Saved median comparison plot to {PATH_TO_SAVE / 'median_search_comparison.png'}"
    )
    return random_results, genetic_results


def process_and_plot_results():
    PARAM_CONFIG = {
        "activation": ("Activation Functions", "activation"),
        "alpha": ("Alpha Values", "alpha"),
        "solver": ("Solvers", "solver"),
        "hidden_neurons_size": ("Hidden Neuron Size", "hidden_neuron_size"),
        "max_iterations": ("Max Iterations", "max_iterations"),
        "scaler": ("Scalers", "scaler"),
    }

    random_results, genetic_results = generate_median_search_comparison_plot(50)

    random_historic_single = random_results[0]["historic"]
    genetic_historic_single = genetic_results[0]["historic"]

    random_running = get_running_max(random_historic_single)
    genetic_running = get_running_max(genetic_historic_single)

    plot.clf()
    plot.plot(range(1, len(random_running) + 1), random_running, label="Random Search")
    plot.plot(
        range(1, len(genetic_running) + 1), genetic_running, label="Genetic Search"
    )
    plot.title("Search Algorithm Performance Comparison")
    plot.xlabel("Number of Evaluations")
    plot.ylabel("Best Accuracy")
    plot.legend()
    plot.savefig(PATH_TO_SAVE / "search_comparison.png")
    plot.close()

    random_historic_all = []
    for res in random_results:
        random_historic_all.extend(res["historic"])

    genetic_historic_all = []
    for res in genetic_results:
        genetic_historic_all.extend(res["historic"])

    random_grouped = group_accuracies_by_param(
        random_historic_all, list(PARAM_CONFIG.keys())
    )
    genetic_grouped = group_accuracies_by_param(
        genetic_historic_all, list(PARAM_CONFIG.keys())
    )

    for param_key, (title, file_prefix) in PARAM_CONFIG.items():
        random_data = random_grouped.get(param_key, {})
        genetic_data = genetic_grouped.get(param_key, {})

        all_keys = sorted(list(set(random_data.keys()) | set(genetic_data.keys())))
        random_means = [calculate_mean(random_data.get(k, [])) for k in all_keys]
        genetic_means = [calculate_mean(genetic_data.get(k, [])) for k in all_keys]

        x = np.arange(len(all_keys))
        width = 0.35

        plot.clf()
        plot.bar(x - width / 2, random_means, width, label="Random Search")
        plot.bar(x + width / 2, genetic_means, width, label="Genetic Search")
        plot.title(f"Mean Accuracy by {title}")
        plot.xlabel(title)
        plot.ylabel("Accuracy")
        plot.xticks(x, all_keys)
        plot.legend()
        plot.savefig(PATH_TO_SAVE / f"{file_prefix}_bar.png")
        plot.close()

        random_medians = [calculate_median(random_data.get(k, [])) for k in all_keys]
        genetic_medians = [calculate_median(genetic_data.get(k, [])) for k in all_keys]

        plot.clf()
        plot.bar(x - width / 2, random_medians, width, label="Random Search")
        plot.bar(x + width / 2, genetic_medians, width, label="Genetic Search")
        plot.title(f"Median Accuracy by {title}")
        plot.xlabel(title)
        plot.ylabel("Accuracy")
        plot.xticks(x, all_keys)
        plot.legend()
        plot.savefig(PATH_TO_SAVE / f"{file_prefix}_median_bar.png")
        plot.close()

        fig, (ax1, ax2) = plot.subplots(1, 2, figsize=(12, 5), sharey=True)
        for param_value, accuracies in random_data.items():
            ax1.hist(accuracies, alpha=0.5, label=param_value, bins=10)
        ax1.set_title("Random Search")
        ax1.set_xlabel("Accuracy")
        ax1.set_ylabel("Frequency")
        ax1.legend()

        for param_value, accuracies in genetic_data.items():
            ax2.hist(accuracies, alpha=0.5, label=param_value, bins=10)
        ax2.set_title("Genetic Search")
        ax2.set_xlabel("Accuracy")
        ax2.legend()

        fig.suptitle(f"Accuracy Distribution by {title}")
        plot.tight_layout()
        plot.savefig(PATH_TO_SAVE / f"{file_prefix}_hist.png")
        plot.close()


if __name__ == "__main__":
    process_and_plot_results()
