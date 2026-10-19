import random
from numbers import Real
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from NeuralNetwork import NeuralNetwork
from training import real_k_fold, FeatureTypes


def _is_numeric(value):
    return isinstance(value, Real) and not isinstance(value, bool)


def _plot_parameter(ax, values, scores, trial_numbers, label, use_log_scale):
    numeric_points = [
        (value, score, trial_number)
        for value, score, trial_number in zip(values, scores, trial_numbers)
        if _is_numeric(value)
    ]
    categorical_points = [
        (str(value), score, trial_number)
        for value, score, trial_number in zip(values, scores, trial_numbers)
        if not _is_numeric(value)
    ]

    scatter = None
    if numeric_points:
        numeric_values, numeric_scores, numeric_trials = zip(*numeric_points)
        scatter = ax.scatter(
            numeric_values,
            numeric_scores,
            c=numeric_trials,
            cmap="viridis",
            alpha=0.8,
        )
        ax.set_xlabel(label if not categorical_points else f"{label} numeric")
        if use_log_scale and min(numeric_values) > 0:
            ax.set_xscale("log")

    if categorical_points:
        categories = []
        for category, _, _ in categorical_points:
            if category not in categories:
                categories.append(category)

        category_positions = {category: index for index, category in enumerate(categories)}
        categorical_values = [category_positions[category] for category, _, _ in categorical_points]
        categorical_scores = [score for _, score, _ in categorical_points]
        categorical_trials = [trial_number for _, _, trial_number in categorical_points]

        category_ax = ax.twiny() if numeric_points else ax
        scatter = category_ax.scatter(
            categorical_values,
            categorical_scores,
            c=categorical_trials,
            cmap="viridis",
            alpha=0.8,
            marker="x" if numeric_points else "o",
        )
        category_ax.set_xlim(-0.5, len(categories) - 0.5)
        category_ax.set_xticks(range(len(categories)))
        category_ax.set_xticklabels(categories)
        category_ax.set_xlabel(f"{label} categories" if numeric_points else label)
        category_ax.grid(False)

    return scatter


def plot_hyperparameter_results(
    results,
    parameter_plots,
    output_path="hyperparameter_results.png",
    show_accuracy_progress=True,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trial_numbers = list(range(1, len(results) + 1))
    val_scores = [result["mean_acc_val"] for result in results]
    balanced_acc_scores = [result["mean_balanced_acc_test"] for result in results]
    best_result = max(results, key=lambda result: result["mean_acc_val"])

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), layout="constrained")
    axes = axes.ravel()
    parameter_axes = axes[:4] if show_accuracy_progress else axes[:5]

    if len(parameter_plots) > len(parameter_axes):
        raise ValueError(f"Can only plot {len(parameter_axes)} hyperparameters.")

    for ax, (param_name, label, use_log_scale) in zip(parameter_axes, parameter_plots):
        values = [result[param_name] for result in results]
        scatter = _plot_parameter(ax, values, val_scores, trial_numbers, label, use_log_scale)
        ax.set_ylabel("Mean validation accuracy")
        ax.set_title(f"{label} vs validation accuracy")
        ax.grid(True, alpha=0.3)

    for ax in parameter_axes[len(parameter_plots):]:
        ax.set_visible(False)

    if show_accuracy_progress:
        best_so_far = []
        balanced_acc_best_so_far = []
        current_best = float("-inf")
        current_balanced_acc_best = float("-inf")
        for score, balanced_acc_score in zip(val_scores, balanced_acc_scores):
            current_best = max(current_best, score)
            current_balanced_acc_best = max(current_balanced_acc_best, balanced_acc_score)
            best_so_far.append(current_best)
            balanced_acc_best_so_far.append(current_balanced_acc_best)

        axes[4].plot(trial_numbers, best_so_far, marker="o", color="tab:green", label="Validation accuracy")
        axes[4].plot(
            trial_numbers,
            balanced_acc_best_so_far,
            marker="o",
            color="tab:orange",
            label="Balanced test accuracy",
        )
        axes[4].set_xlabel("Trial")
        axes[4].set_ylabel("Best accuracy")
        axes[4].set_title("Best accuracy found")
        axes[4].grid(True, alpha=0.3)
        axes[4].legend()

    confusion_matrix = best_result["total_cm"]
    matrix_plot = axes[5].imshow(confusion_matrix, cmap="Blues")
    axes[5].set_xlabel("Predicted label")
    axes[5].set_ylabel("True label")
    axes[5].set_title("Best model confusion matrix")
    axes[5].set_xticks(range(confusion_matrix.shape[1]))
    axes[5].set_yticks(range(confusion_matrix.shape[0]))
    for row in range(confusion_matrix.shape[0]):
        for col in range(confusion_matrix.shape[1]):
            axes[5].text(
                col,
                row,
                int(confusion_matrix[row, col]),
                ha="center",
                va="center",
                color="white" if confusion_matrix[row, col] > confusion_matrix.max() / 2 else "black",
            )

    fig.colorbar(scatter, ax=parameter_axes[:len(parameter_plots)], label="Trial number")
    fig.colorbar(matrix_plot, ax=axes[5], label="Count")
    fig.savefig(output_path, dpi=200)
    plt.show()


def get_neural_hyperparameters():
    budget = 100

    dropout_rates = [random.uniform(0.1, 0.6) for _ in range(budget)]
    weight_decays = [10 ** random.uniform(-5, -2) for _ in range(budget)]
    learning_rates = [10 ** random.uniform(-4, -2) for _ in range(budget)]
    batch_sizes = [random.choice([8, 16, 32, 64]) for _ in range(budget)]
    network_sizes = [random.choice([32, 16, 8]) for _ in range(budget)]
    results = []
    for i in range(budget):
        model = NeuralNetwork(
            [38, network_sizes[i], int(network_sizes[i]/2), 4],
            epochs=100,
            batch_size=batch_sizes[i],
            lr=learning_rates[i],
            weight_decay=weight_decays[i],
            dropout=dropout_rates[i],
        )
        scores = real_k_fold(model, FeatureTypes.PCA)
        results.append(
            {
                "dropout": dropout_rates[i],
                "weight_decay": weight_decays[i],
                "learning_rate": learning_rates[i],
                "batch_size": batch_sizes[i],
                "network_size": network_sizes[i],
                **scores,
            }
        )
        print(
            f"Trial {i + 1}/{budget}: "
            f"val_acc={scores['mean_acc_val']:.4f}, "
            f"balanced_acc={scores['mean_balanced_acc_test']:.4f}, "
            f"test_acc={scores['mean_acc_test']:.4f}, "
            f"dropout={dropout_rates[i]:.4f}, "
            f"weight_decay={weight_decays[i]:.6f}, "
            f"lr={learning_rates[i]:.6f}, "
            f"batch_size={batch_sizes[i]}, "
            f"network_size={network_sizes[i]}"
        )

    best_result = max(results, key=lambda result: result["mean_acc_val"])
    print("\nBest hyperparameters by validation accuracy:")
    print(f"Validation accuracy: {best_result['mean_acc_val']:.4f}")
    print(f"Balanced accuracy: {best_result['mean_balanced_acc_test']:.4f}")
    print(f"Test accuracy: {best_result['mean_acc_test']:.4f}")
    print(f"Macro F1 score: {best_result['mean_macro_f1_test']:.4f}")
    print(f"Dropout: {best_result['dropout']:.4f}")
    print(f"Weight decay: {best_result['weight_decay']:.6f}")
    print(f"Learning rate: {best_result['learning_rate']:.6f}")
    print(f"Batch size: {best_result['batch_size']}")
    print(f"Network size: {best_result['network_size']}")

    neural_parameter_plots = [
        ("dropout", "Dropout rate", False),
        ("weight_decay", "Weight decay", True),
        ("learning_rate", "Learning rate", True),
        ("batch_size", "Batch size", False),
        ("network_size", "Network size", False),
    ]
    plot_hyperparameter_results(
        results,
        neural_parameter_plots,
        show_accuracy_progress=False,
    )
    return results


def get_forest_hyperparameters():
    budget = 100
    number_of_trees = [random.randint(25, 200) for _ in range(budget)]
    max_depth = [random.randint(4, 12) for _ in range(budget)]
    min_samples_leaf = [random.randint(1, 5) for _ in range(budget)]
    min_samples_split = [random.randint(2, 10) for _ in range(budget)]
    results = []

    for i in range(budget):
        model = RandomForestClassifier(
            n_estimators=number_of_trees[i],
            max_depth=max_depth[i],
            min_samples_leaf=min_samples_leaf[i],
            min_samples_split=min_samples_split[i],
            random_state=31,
            n_jobs=-1,
        )
        scores = real_k_fold(model, FeatureTypes.PCA)
        print(
            f"Trial {i + 1}/{budget}: "
            f"val_acc={scores['mean_acc_val']:.4f}, "
            f"balanced_acc={scores['mean_balanced_acc_test']:.4f}, "
            f"test_acc={scores['mean_acc_test']:.4f}"
        )
        results.append(
            {
                "n_estimators": number_of_trees[i],
                "max_depth": max_depth[i],
                "min_samples_leaf": min_samples_leaf[i],
                "min_samples_split": min_samples_split[i],
                **scores,
            }
        )
        print(
            f"Trial {i + 1}/{budget}: "
            f"val_acc={scores['mean_acc_val']:.4f}, "
            f"balanced_acc={scores['mean_balanced_acc_test']:.4f}, "
            f"test_acc={scores['mean_acc_test']:.4f}, "
            f"n_estimators={number_of_trees[i]}, "
            f"max_depth={max_depth[i]}, "
            f"min_samples_leaf={min_samples_leaf[i]}, "
            f"min_samples_split={min_samples_split[i]}"
        )

    best_result = max(results, key=lambda result: result["mean_acc_val"])
    print("\nBest random forest hyperparameters by validation accuracy:")
    print(f"Validation accuracy: {best_result['mean_acc_val']:.4f}")
    print(f"Balanced accuracy: {best_result['mean_balanced_acc_test']:.4f}")
    print(f"Test accuracy: {best_result['mean_acc_test']:.4f}")
    print(f"Macro F1 score: {best_result['mean_macro_f1_test']:.4f}")
    print(f"Number of trees: {best_result['n_estimators']}")
    print(f"Max depth: {best_result['max_depth']}")
    print(f"Min samples leaf: {best_result['min_samples_leaf']}")
    print(f"Min samples split: {best_result['min_samples_split']}")

    forest_parameter_plots = [
        ("n_estimators", "Number of trees", False),
        ("max_depth", "Max depth", False),
        ("min_samples_leaf", "Min samples leaf", False),
        ("min_samples_split", "Min samples split", False),
    ]
    plot_hyperparameter_results(
        results,
        forest_parameter_plots,
        output_path="figures/scaled_forest_hyperparameter_results.png",
    )
    return results


def get_svm_hyperparameters():
    budget = 100
    c_values = [10 ** random.uniform(-3, 3) for _ in range(budget)]
    gamma_values = [random.choice(["scale", "auto", 10 ** random.uniform(-4, 1), 10 ** random.uniform(-4, 1)]) for _ in range(budget)]
    kernels = [random.choice(["linear", "rbf", "poly"]) for _ in range(budget)]
    degrees = [random.randint(2, 5) for _ in range(budget)]
    results = []

    for i in range(budget):
        model = SVC(
            C=c_values[i],
            gamma=gamma_values[i],
            kernel=kernels[i],
            degree=degrees[i],
            cache_size=1000,
        )
        scores = real_k_fold(model, FeatureTypes.PCA)
        results.append(
            {
                "C": c_values[i],
                "gamma": gamma_values[i],
                "kernel": kernels[i],
                "degree": degrees[i],
                **scores,
            }
        )
        print(
            f"Trial {i + 1}/{budget}: "
            f"val_acc={scores['mean_acc_val']:.4f}, "
            f"balanced_acc={scores['mean_balanced_acc_test']:.4f}, "
            f"test_acc={scores['mean_acc_test']:.4f}, "
            f"C={c_values[i]:.6f}, "
            f"gamma={gamma_values[i]}, "
            f"kernel={kernels[i]}, "
            f"degree={degrees[i]}"
        )

    best_result = max(results, key=lambda result: result["mean_acc_val"])
    print("\nBest SVM hyperparameters by validation accuracy:")
    print(f"Validation accuracy: {best_result['mean_acc_val']:.4f}")
    print(f"Balanced accuracy: {best_result['mean_balanced_acc_test']:.4f}")
    print(f"Test accuracy: {best_result['mean_acc_test']:.4f}")
    print(f"Macro F1 score: {best_result['mean_macro_f1_test']:.4f}")
    print(f"C: {best_result['C']:.6f}")
    print(f"Gamma: {best_result['gamma']}")
    print(f"Kernel: {best_result['kernel']}")
    print(f"Degree: {best_result['degree']}")

    svm_parameter_plots = [
        ("C", "C", True),
        ("gamma", "Gamma", True),
        ("kernel", "Kernel", False),
        ("degree", "Degree", False),
    ]
    plot_hyperparameter_results(
        results,
        svm_parameter_plots,
        output_path="figures/scaled_svm_hyperparameter_results.png",
    )
    return results


if __name__ == "__main__":
    get_neural_hyperparameters()
