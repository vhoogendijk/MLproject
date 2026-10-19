from pathlib import Path
import runpy

import numpy as np
import matplotlib.pyplot as plt


# only using the training set here
def plot_run_features(raw, scaled, labels, names, run, output):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    # turning this into a list makes it easier to find each feature column
    names = list(names)
    classes = list(gesture_labels.items())
    titles = [name.title() for name, _ in classes]
    colors = ["tab:blue", "tab:orange", "tab:purple", "tab:pink"]

    # saving both formats (pdf, png)
    def save(fig, suffix):
        fig.savefig(output / f"run_{run}_{suffix}.png", dpi=300, bbox_inches="tight")
        fig.savefig(output / f"run_{run}_{suffix}.pdf", bbox_inches="tight")
        plt.show()  
        plt.close(fig)

    # centroid and bandwidth
    # each recording gives one mean value, then the boxes show the spread per gesture
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
    for ax, column, title in zip(
        axes, ["centroid_mean", "bandwidth_mean"],
        ["Mean Doppler centroid", "Mean RMS Doppler bandwidth"]
    ):
        column_index = names.index(column)
        values = []
        for gesture, label in classes:
            gesture_values = raw[labels == label, column_index]
            values.append(gesture_values)
        ax.boxplot(values, tick_labels=titles, widths=0.5,
                   medianprops={"color": "tab:orange"},
                   flierprops={"markersize": 3})
        ax.set_title(title)
        ax.set_ylabel("Normalised Doppler coordinate")
        ax.tick_params(labelsize=9)
    fig.suptitle(f"Centroid and bandwidth - run {run} training set")
    fig.tight_layout()
    save(fig, "centroid_bandwidth")

    # envelopes
    # finding the positive and negative columns in the feature table
    # the recordings were already resampled to the same number of time points
    positive = [i for i, name in enumerate(names) if name.startswith("envelope_positive_")]
    negative = [i for i, name in enumerate(names) if name.startswith("envelope_negative_")]
    # using relative time since the recordings can have different lengths
    time = np.linspace(0, 1, len(positive))
    fig, axes = plt.subplots(2, 2, figsize=(9, 5.5), sharex=True, sharey=True)
    for ax, (gesture, label) in zip(axes.flat, classes):
        # take all the training recordings for this gesture
        selected = labels == label
        for indices, color in [(positive, "tab:orange"), (negative, "tab:blue")]:
            # the line is the median and the shaded part is the middle 50%
            lower, median, upper = np.percentile(raw[selected][:, indices], [25, 50, 75], axis=0)
            ax.fill_between(time, lower, upper, color=color, alpha=0.2, linewidth=0)
            ax.plot(time, median, color=color, linewidth=1.5)
        ax.axhline(0, color="0.6", linewidth=0.6)
        ax.set_title(f"{gesture.title()} (n = {selected.sum()})", fontsize=10)
        ax.set_xlim(0, 1)
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Envelope features - run {run} training set", y=0.99)
    # one shared legend keeps the four small plots less crowded
    fig.legend(handles=[Line2D([], [], color="tab:orange", label="Positive median"),
                        Line2D([], [], color="tab:blue", label="Negative median"),
                        Patch(facecolor="0.7", alpha=0.4, label="25th-75th percentile")],
               loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=3, frameon=False, fontsize=9)
    fig.supxlabel("Relative recording time")
    fig.supylabel("Normalised Doppler coordinate")
    fig.tight_layout(rect=[0.03, 0.03, 1, 0.86])
    save(fig, "envelopes")

    # chebyshev moments
    # these were scaled using the training set, with each coefficient scaled separately
    # this makes it easier to compare moments that have very different raw values
    indices = [i for i, name in enumerate(names) if name.startswith("chebyshev_")]
    # shorter labels so all 28 moment orders fit on the x axis
    ticks = []
    for index in indices:
        short_name = names[index].removeprefix("chebyshev_d").replace("_c", ", ")
        ticks.append(short_name)
    positions = np.arange(len(indices))
    medians = []
    for gesture, label in classes:
        gesture_moments = scaled[labels == label][:, indices]
        medians.append(np.median(gesture_moments, axis=0))
    # same y limits for all four gestures, with a little space above the bars
    limit = max(1.0, 1.15 * np.max(np.abs(medians)))
    fig, axes = plt.subplots(4, 1, figsize=(10, 6), sharex=True, sharey=True)
    for ax, (gesture, label), color, values in zip(axes, classes, colors, medians):
        ax.bar(positions, values, color=color, width=0.8)
        ax.axhline(0, color="0.5", linewidth=0.6)
        ax.set_ylim(-limit, limit)
        ax.set_ylabel(gesture.title(), rotation=0, labelpad=35, color=color, fontweight="bold")
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xticks(positions, [f"({tick})" for tick in ticks], rotation=90, fontsize=8)
    axes[-1].set_xlabel("Moment orders (Doppler, cadence)")
    fig.suptitle(f"Chebyshev moments - run {run} training set\nBar height = median standardized coefficient")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save(fig, "chebyshev")


def main():
    global gesture_labels
    # the preprocessing file needs to be in the same folder as this one
    preprocessing_file = Path(__file__).resolve().with_name("preprocessing.py")
    print("First running the preprocessing")
    # run it first so the saved feature files match the current preprocessing
    result = runpy.run_path(str(preprocessing_file), run_name="__main__")
    # getting the labels and output folder from that file
    gesture_labels = result["gesture_labels"]
    output = result["output"]

    # loading and plotting one run at a time
    for run in range(1, len(result["splits"]) + 1):
        # raw values for the boxes and envelopes, scaled values for the moments
        with np.load(output / f"run_{run}_raw_features.npz") as raw, np.load(
            output / f"run_{run}_scaled_features.npz"
        ) as scaled:
            plot_run_features(raw["X_train"], scaled["X_train"], raw["y_train"],
                              raw["feature_names"], run, output)

        # also show the pca plot if pca was switched on in preprocessing
        if result["use_pca"]:
            with np.load(output / f"run_{run}_pca_features.npz") as pca:
                values = pca["X_train"]
                # need at least two components for a 2d scatter plot
                if values.shape[1] < 2:
                    continue
                fig, ax = plt.subplots(figsize=(7, 5))
                for gesture, label in gesture_labels.items():
                    selected = pca["y_train"] == label
                    ax.scatter(values[selected, 0], values[selected, 1],
                               label=gesture, s=16, alpha=0.6)
                # showing how much training variance each axis explains
                ratios = pca["explained_variance_ratio"]
                ax.set_xlabel(f"PC1 ({ratios[0]:.1%} of training variance)")
                ax.set_ylabel(f"PC2 ({ratios[1]:.1%} of training variance)")
                ax.set_title(f"Run {run}: training features")
                ax.legend()
                fig.tight_layout()
                fig.savefig(output / f"run_{run}_pca.png", dpi=300, bbox_inches="tight")
                fig.savefig(output / f"run_{run}_pca.pdf", bbox_inches="tight")
                plt.show()
                plt.close(fig)


if __name__ == "__main__":
    main()
