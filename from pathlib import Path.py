from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# creating the output folder
root = Path(r"C:\Users\SID-DRW\Downloads\WICOS - Copy")
output = root / "Project" / "Project_Output"
output.mkdir(parents=True, exist_ok=True)

files = sorted(root.rglob("Person*g*s*.csv"))
print(f"Found {len(files)} recordings")


# change i to change which recording you look at
i = 600
spectrogram = np.loadtxt(files[i], delimiter=",", dtype=np.float32)
gesture = files[i].parent.name
person = files[i].parent.parent.name

print("File:", files[i].name)
print("Shape:", spectrogram.shape)
print("All values finite:", np.isfinite(spectrogram).all())

plt.figure(figsize=(9, 4))
plt.imshow(spectrogram, aspect="auto", origin="lower", cmap="rainbow")
plt.xlabel("Time bin")
plt.ylabel("Doppler bin")
plt.colorbar(label="Level (dB)")
plt.title(f"{person} - {gesture}")
plt.tight_layout()
plt.show()


# four training people, one validation person and one test person.
# without validation: five training people and one test person.
def make_person_splits(people, use_validation=True):
    splits = []

    for i in range(len(people)):
        # Use every person once for testing.
        test_person = people[i]

        if use_validation:
            validation_person = people[(i + 1) % len(people)]
        else:
            validation_person = None

        train_people = [
            person for person in people
            if person != test_person and person != validation_person
        ]

        splits.append((train_people, validation_person, test_person))

    return splits


# mapping the gestures to labels
gesture_labels = {
    "wave": 0,
    "pinch": 1,
    "swipe": 2,
    "click": 3,
}

recordings = []
rows = []
duplicates = []

for file in files:
    # checking for duplicates / invalid values 
    spectrogram = np.loadtxt(file, delimiter=",", dtype=np.float64)

    if not np.isfinite(spectrogram).all():
        raise ValueError(f"Invalid values in {file}")

    is_duplicate = False

    for saved_recording in recordings:
        if np.array_equal(spectrogram, saved_recording):
            is_duplicate = True
            break

    if is_duplicate:
        duplicates.append(file)
        continue

    recordings.append(spectrogram)

    # get the gesture and person from the folder names.
    gesture = file.parent.name
    person = file.parent.parent.name.replace("Person", "").strip()

    rows.append({
        "file": str(file),
        "person": person,
        "gesture": gesture,
        "label": gesture_labels[gesture],
    })


# putting the data into a df
data = pd.DataFrame(rows)
y = data["label"].to_numpy()

assert len(recordings) == len(data) == len(y)

print(data.head())
print("Recordings kept:", len(recordings))
print("Duplicates excluded:", len(duplicates))


# normalising the spectogram between 0 and 1 
def normalise_spectrogram(spectrogram, minimum_db):
    normalized = (spectrogram - minimum_db) / (0 - minimum_db)
    normalized = np.clip(normalized, 0, 1)

    target_columns = 512
    columns = normalized.shape[1]

    if columns > target_columns:
        raise ValueError("Recording has more than 512 time columns.")

    # zero pad shorter recordings
    padded = np.pad(
        normalized,
        ((0, 0), (0, target_columns - columns)),
        mode="constant",
        constant_values=0,
    )

    image = Image.fromarray(padded.astype(np.float32))
    image = image.resize((128, 128), resample=Image.Resampling.BOX)

    return np.array(image, dtype=np.float32)


# paper 1: positive and negative envelopes
def get_envelopes(power, doppler, fraction=0.01, noise_multiplier=20.0):
    positive = np.zeros(power.shape[1])
    negative = np.zeros(power.shape[1])

    frame_power = power.sum(axis=0)
    positive_rows = doppler > 0
    negative_rows = doppler < 0

    for column in range(power.shape[1]):
        # ignore columns with very little power relative to this recording
        if (
            frame_power[column] <= 0
            or frame_power[column] < 0.01 * frame_power.max()
        ):
            continue

        # reduce the chance of background bins becoming envelope edges
        # does risk losing a weak gesture edge
        noise_limit = noise_multiplier * np.median(power[:, column])

        for side, envelope in [
            (positive_rows, positive),
            (negative_rows, negative),
        ]:
            side_power = power[side, column]
            side_doppler = doppler[side]
            side_total = side_power.sum()

            if side_total < 0.01 * frame_power[column]:
                continue

            threshold = max(fraction * side_total, noise_limit)
            selected = side_doppler[side_power >= threshold]

            if len(selected) > 0:
                # keep the qualifying Doppler bin furthest from zero.
                envelope[column] = selected[np.argmax(np.abs(selected))]

    return positive, negative


# doppler centroid and width
def get_centroid_bandwidth(power, doppler):
    frame_power = power.sum(axis=0)

    # no division by zero if frame_power is zero, the centroid and bandwidth will be NaN.
    denominator = np.where(frame_power > 0, frame_power, 1)

    # calculating the centroid
    centroid = (doppler[:, None] * power).sum(axis=0) / denominator

    # calculating the BW
    squared_distance = (doppler[:, None] - centroid[None, :]) ** 2
    bandwidth = np.sqrt(
        (squared_distance * power).sum(axis=0) / denominator
    )

    return centroid, bandwidth


# discrete chebyshev polynomials
def chebyshev_polynomials(size, order):
    if size <= order:
        raise ValueError(
            "The size is smaller than the order of the polynomials"
        )

    x = np.arange(size, dtype=np.float64)
    polynomials = np.zeros((order + 1, size))

    polynomials[0] = 1

    if order >= 1:
        polynomials[1] = (2 * x + 1 - size) / size

    # recurrence for the scaled polynomials t_n(x) / size**n.
    for n in range(2, order + 1):
        first = (2 * n - 1) * (2 * x + 1 - size) / size
        second = (n - 1) * (1 - ((n - 1) / size) ** 2)

        polynomials[n] = (
            first * polynomials[n - 1]
            - second * polynomials[n - 2]
        ) / n

    return polynomials


# calculate Chebyshev moments of the cadence–velocity diagram.
def get_chebyshev_features(power, order=6):
    amplitude = np.sqrt(power)
    cvd = np.abs(np.fft.fft(amplitude, axis=1))

    doppler_basis = chebyshev_polynomials(cvd.shape[0], order)
    cadence_basis = chebyshev_polynomials(cvd.shape[1], order)

    doppler_norm = (doppler_basis ** 2).sum(axis=1)
    cadence_norm = (cadence_basis ** 2).sum(axis=1)

    # project the CVD onto the polynomial patterns
    moments = doppler_basis @ cvd @ cadence_basis.T
    moments = moments / doppler_norm[:, None] / cadence_norm[None, :]

    features = {}

    # store the 28 moments used in the feature table
    for total_order in range(order + 1):
        for doppler_order in range(total_order + 1):
            cadence_order = total_order - doppler_order
            name = f"chebyshev_d{doppler_order}_c{cadence_order}"
            features[name] = moments[doppler_order, cadence_order]

    return features


# combine the three papers to extract a single feature vector for each recording
def extract_features(
    spectrogram,
    envelope_points=32,
    envelope_fraction=0.01,
    noise_multiplier=20.0,
    chebyshev_order=6,
):
    # dB = 10*log10(power)
    power = 10.0 ** (spectrogram / 10.0)
    doppler = np.linspace(-1, 1, spectrogram.shape[0])

    positive, negative = get_envelopes(
        power,
        doppler,
        envelope_fraction,
        noise_multiplier,
    )

    # resample both envelopes to the same length
    old_time = np.linspace(0, 1, spectrogram.shape[1])
    new_time = np.linspace(0, 1, envelope_points)

    positive = np.interp(new_time, old_time, positive)
    negative = np.interp(new_time, old_time, negative)

    features = {}

    for i in range(envelope_points):
        features[f"envelope_positive_{i:02d}"] = positive[i]

    for i in range(envelope_points):
        features[f"envelope_negative_{i:02d}"] = negative[i]

    centroid, bandwidth = get_centroid_bandwidth(power, doppler)
    valid_columns = power.sum(axis=0) > 0

    features["centroid_mean"] = centroid[valid_columns].mean()
    features["bandwidth_mean"] = bandwidth[valid_columns].mean()

    features.update(get_chebyshev_features(power, chebyshev_order))

    return features


# extract featrures once
feature_rows = []

for spectrogram in recordings:
    feature_rows.append(extract_features(spectrogram))

features = pd.DataFrame(feature_rows)
X_features = features.to_numpy(dtype=np.float64)
feature_names = features.columns.to_numpy(dtype=str)

# 64 envelope values + 2 centroid/bandwidth means + 28 moments.
assert X_features.shape == (len(recordings), 94)
assert np.isfinite(X_features).all()

print("Feature matrix shape:", X_features.shape)

# save the features
pd.concat([data, features], axis=1).to_csv(
    output / "features_raw.csv",
    index=False,
)

# optional since the first paper mentioned that their method works better than using pca
use_pca = True


# person splits
people = sorted(data["person"].unique())
splits = make_person_splits(people, use_validation=True)

for run, (train_people, validation_person, test_person) in enumerate(
    splits, start=1
):
    train_mask = data["person"].isin(train_people).to_numpy()
    test_mask = (data["person"] == test_person).to_numpy()

    if validation_person is not None:
        validation_mask = (
            data["person"] == validation_person
        ).to_numpy()
    else:
        validation_mask = np.zeros(len(data), dtype=bool)

    # estimating minimum_db for every run
    training_values = []

    for i in range(len(recordings)):
        if train_mask[i]:
            values = recordings[i].ravel()

            positions = np.linspace(
                0, len(values) - 1, 2048, dtype=int
            )
            training_values.append(values[positions])

    minimum_db = np.percentile(
        np.concatenate(training_values), 1
    )

    images = []

    for spectrogram in recordings:
        images.append(normalise_spectrogram(spectrogram, minimum_db))

    X = np.array(images, dtype=np.float32)

    X_train = X[train_mask]
    y_train = y[train_mask]

    X_validation = X[validation_mask]
    y_validation = y[validation_mask]

    X_test = X[test_mask]
    y_test = y[test_mask]

    print(f"\nRun {run}")
    print("Training people:", train_people)
    print("Validation person:", validation_person)
    print("Test person:", test_person)
    print("Training image shape:", X_train.shape)
    print("Lower dB limit:", minimum_db)

    np.savez_compressed(
        output / f"run_{run}_images.npz",
        X_train=X_train,
        y_train=y_train,
        X_validation=X_validation,
        y_validation=y_validation,
        X_test=X_test,
        y_test=y_test,
        train_people=np.array(train_people),
        validation_person=(
            validation_person if validation_person is not None else ""
        ),
        test_person=test_person,
        minimum_db=minimum_db,
    )

    features_train = X_features[train_mask]
    features_validation = X_features[validation_mask]
    features_test = X_features[test_mask]

    np.savez_compressed(
        output / f"run_{run}_raw_features.npz",
        X_train=features_train,
        y_train=y_train,
        X_validation=features_validation,
        y_validation=y_validation,
        X_test=features_test,
        y_test=y_test,
        feature_names=feature_names,
        train_indices=np.flatnonzero(train_mask),
        validation_indices=np.flatnonzero(validation_mask),
        test_indices=np.flatnonzero(test_mask),
    )

    scaler = StandardScaler()
    features_train_scaled = scaler.fit_transform(features_train)
    features_test_scaled = scaler.transform(features_test)

    if validation_person is not None:
        features_validation_scaled = scaler.transform(
            features_validation
        )
    else:
        features_validation_scaled = np.empty(
            (0, X_features.shape[1])
        )

    np.savez_compressed(
        output / f"run_{run}_scaled_features.npz",
        X_train=features_train_scaled,
        y_train=y_train,
        X_validation=features_validation_scaled,
        y_validation=y_validation,
        X_test=features_test_scaled,
        y_test=y_test,
        feature_names=feature_names,
        scaler_mean=scaler.mean_,
        scaler_scale=scaler.scale_,
        train_indices=np.flatnonzero(train_mask),
        validation_indices=np.flatnonzero(validation_mask),
        test_indices=np.flatnonzero(test_mask),
    )

    if use_pca:
        pca = PCA(n_components=0.95, svd_solver="full")

        features_train_pca = pca.fit_transform(features_train_scaled)
        features_test_pca = pca.transform(features_test_scaled)

        if validation_person is not None:
            features_validation_pca = pca.transform(
                features_validation_scaled
            )
        else:
            features_validation_pca = np.empty(
                (0, pca.n_components_)
            )

        print("Original feature count:", features_train.shape[1])
        print("PCA component count:", pca.n_components_)
        print(
            "Training variance retained:",
            f"{pca.explained_variance_ratio_.sum():.1%}",
        )

        np.savez_compressed(
            output / f"run_{run}_pca_features.npz",
            X_train=features_train_pca,
            y_train=y_train,
            X_validation=features_validation_pca,
            y_validation=y_validation,
            X_test=features_test_pca,
            y_test=y_test,
            feature_names=feature_names,
            scaler_mean=scaler.mean_,
            scaler_scale=scaler.scale_,
            pca_mean=pca.mean_,
            pca_components=pca.components_,
            explained_variance_ratio=pca.explained_variance_ratio_,
            train_indices=np.flatnonzero(train_mask),
            validation_indices=np.flatnonzero(validation_mask),
            test_indices=np.flatnonzero(test_mask),
        )

        # visualizing training features
        if pca.n_components_ >= 2:
            plt.figure(figsize=(7, 5))

            for gesture, label in gesture_labels.items():
                selected = y_train == label

                plt.scatter(
                    features_train_pca[selected, 0],
                    features_train_pca[selected, 1],
                    label=gesture,
                    s=16,
                    alpha=0.6,
                )

            ratios = pca.explained_variance_ratio_
            plt.xlabel(f"PC1 ({ratios[0]:.1%} of training variance)")
            plt.ylabel(f"PC2 ({ratios[1]:.1%} of training variance)")
            plt.title(f"Run {run}: training features")
            plt.legend()
            plt.tight_layout()
            plt.savefig(
                output / f"run_{run}_pca.png",
                dpi=150,
                bbox_inches="tight",
            )
            plt.show()
            plt.close()

# choose the recording to be visualised
i = 600

example = np.loadtxt(files[i], delimiter=",", dtype=np.float64)
gesture = files[i].parent.name
person = files[i].parent.parent.name

power = 10.0 ** (example / 10.0)

doppler = np.linspace(-1, 1, example.shape[0])
time = np.arange(example.shape[1])

#paper 1
positive, negative = get_envelopes(power, doppler)

sample_times = np.linspace(0, example.shape[1] - 1, 32)
positive_samples = np.interp(sample_times, time, positive)
negative_samples = np.interp(sample_times, time, negative)

fig, ax = plt.subplots(figsize=(10, 5))

background = ax.imshow(
    example,
    origin="lower",
    aspect="auto",
    cmap="Greys",
    extent=[-0.5, example.shape[1] - 0.5, -1, 1],
)

ax.plot(time, positive, color="red", label="Positive envelope")
ax.plot(time, negative, color="blue", label="Negative envelope")

# Mark the actual sampled feature values.
ax.scatter(sample_times, positive_samples, color="red", s=20)
ax.scatter(sample_times, negative_samples, color="blue", s=20)

ax.set_xlabel("Time bin")
ax.set_ylabel("Relative Doppler coordinate")
ax.set_title(
    f"Paper 1: envelopes — {person}, {gesture}\n"
    "32 sampled values per curve = 64 features"
)
ax.legend()
fig.colorbar(background, ax=ax, label="Level (dB)")

plt.tight_layout()
plt.savefig(output / "paper_1_envelopes.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close(fig)


#paper 2

centroid, bandwidth = get_centroid_bandwidth(power, doppler)

valid_columns = power.sum(axis=0) > 0
centroid_mean = centroid[valid_columns].mean()
bandwidth_mean = bandwidth[valid_columns].mean()

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(time, centroid, color="blue")
axes[0].axhline(
    centroid_mean,
    color="red",
    linestyle="--",
    label=f"Mean = {centroid_mean:.5f}",
)
axes[0].set_title("Doppler centroid")
axes[0].set_xlabel("Time bin")
axes[0].set_ylabel("Relative Doppler units")
axes[0].legend()

axes[1].plot(time, bandwidth, color="blue")
axes[1].axhline(
    bandwidth_mean,
    color="red",
    linestyle="--",
    label=f"Mean = {bandwidth_mean:.5f}",
)
axes[1].set_title("RMS Doppler bandwidth")
axes[1].set_xlabel("Time bin")
axes[1].set_ylabel("Relative Doppler units")
axes[1].legend()

fig.suptitle(f"Paper 2: centroid and bandwidth — {person}, {gesture}")

plt.tight_layout()
plt.savefig(
    output / "paper_2_centroid_bandwidth.png",
    dpi=150,
    bbox_inches="tight",
)
plt.show()
plt.close(fig)

print("Centroid mean:", centroid_mean)
print("Bandwidth mean:", bandwidth_mean)


# paper 3

amplitude = np.sqrt(power)
cvd = np.abs(np.fft.fft(amplitude, axis=1))

moment_features = get_chebyshev_features(power, order=6)

moment_names = list(moment_features.keys())
moment_values = list(moment_features.values())

short_names = [
    name.replace("chebyshev_", "")
    for name in moment_names
]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

cvd_display = np.log1p(np.fft.fftshift(cvd, axes=1))

cvd_plot = axes[0].imshow(
    cvd_display,
    origin="lower",
    aspect="auto",
    cmap="viridis",
)
axes[0].set_title("Cadence–velocity diagram")
axes[0].set_xlabel("Shifted cadence-bin index")
axes[0].set_ylabel("Doppler-bin index")

fig.colorbar(
    cvd_plot,
    ax=axes[0],
    label="log(1 + CVD magnitude)",
)

positions = np.arange(len(moment_values))

axes[1].bar(positions, moment_values, color="purple")
axes[1].set_xticks(positions)
axes[1].set_xticklabels(short_names, rotation=90, fontsize=8)
axes[1].axhline(0, color="black", linewidth=0.5)
axes[1].set_title("28 Chebyshev moments")
axes[1].set_xlabel("Moment orders: d = Doppler, c = cadence")
axes[1].set_ylabel("Moment coefficient")

fig.suptitle(f"Paper 3: Chebyshev features — {person}, {gesture}")

plt.tight_layout()
plt.savefig(output / "paper_3_chebyshev.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close(fig)



# ###################comparing all#######################

# # change the run you want to analyse (1-6)
# analysis_run = 1
# if not 1 <= analysis_run <= len(splits):
#     raise ValueError(f"Choose an analysis run from 1 to {len(splits)}.")
# analysis_train_people, analysis_validation_person, analysis_test_person = splits[analysis_run - 1]
# analysis_mask = data["person"].isin(analysis_train_people).to_numpy()

# # choose a training person for analysis 1 and a gesture for analysis 2.
# example_person = analysis_train_people[0]
# example_gesture = "wave"
# gestures = list(gesture_labels)

# if example_person not in analysis_train_people:
#     raise ValueError("Choose an example person from this run's training people.")
# if example_gesture not in gestures:
#     raise ValueError("Choose wave, pinch, swipe or click.")
# assert len(recordings) == len(data) == len(features)

# analysis_output = output / f"analysis_run_{analysis_run}"
# analysis_output.mkdir(parents=True, exist_ok=True)

# # All comparison spectrograms use the same colour scale.
# with np.load(output / f"run_{analysis_run}_images.npz") as saved_run:
#     display_minimum_db = float(saved_run["minimum_db"])

# moment_columns = [name for name in features.columns if name.startswith("chebyshev_")]


# def plot_feature_comparison(indices, row_names, title, filename):
#     fig, axes = plt.subplots(
#         len(indices), 3,
#         figsize=(15, 3.2 * len(indices)),
#         squeeze=False,
#         sharex="col",
#         sharey="col",
#         layout="constrained",
#     )

#     for row, index in enumerate(indices):
#         spectrogram = recordings[index]
#         power = 10.0 ** (spectrogram / 10.0)
#         doppler = np.linspace(-1, 1, spectrogram.shape[0])

#         time = np.linspace(0, 1, spectrogram.shape[1])
#         positive, negative = get_envelopes(power, doppler)
#         centroid, bandwidth = get_centroid_bandwidth(power, doppler)

#         # Paper 1
#         ax = axes[row, 0]
#         ax.imshow(
#             spectrogram, origin="lower", aspect="auto", cmap="Greys",
#             extent=[0, 1, -1, 1], vmin=display_minimum_db, vmax=0,
#         )
#         ax.plot(time, positive, color="tab:red", label="Positive envelope")
#         ax.plot(time, negative, color="tab:blue", label="Negative envelope")
#         ax.set_title(f"{row_names[row]}: envelopes")
#         ax.set_xlabel("Relative gesture time")
#         ax.set_ylabel("Relative Doppler")
#         ax.set_ylim(-1, 1)

#         # Paper 2
#         ax = axes[row, 1]
#         ax.plot(time, centroid, label="Centroid")
#         ax.plot(time, bandwidth, label="RMS bandwidth")
#         ax.set_title(f"{row_names[row]}: centroid / bandwidth")
#         ax.set_xlabel("Relative gesture time")
#         ax.set_ylabel("Relative Doppler units")

#         # Paper 3
#         ax = axes[row, 2]
#         values = features.iloc[index][moment_columns].to_numpy(dtype=float)
#         ax.bar(np.arange(len(values)), values, color="tab:purple")
#         ax.axhline(0, color="black", linewidth=0.5)
#         ax.set_title(f"{row_names[row]}: Chebyshev moments")
#         ax.set_xlabel("Moment index (0 to 27)")
#         ax.set_ylabel("Moment coefficient")

#     axes[0, 0].legend(fontsize=8)
#     axes[0, 1].legend(fontsize=8)
#     fig.suptitle(title, fontsize=15)
#     fig.canvas.draw()
#     fig.savefig(analysis_output / filename, dpi=150, bbox_inches="tight")
#     plt.show()
#     plt.close(fig)


# # 1. Compare all four gestures from one training person.
# # Select the first retained recording in each group, without choosing by appearance.
# # These examples illustrate the features; they do not represent all recordings.
# gesture_indices = []

# for gesture in gestures:
#     matches = np.flatnonzero(
#         analysis_mask
#         & (data["person"].to_numpy() == example_person)
#         & (data["gesture"].to_numpy() == gesture)
#     )
#     if len(matches) == 0:
#         raise ValueError(f"No {gesture} recordings found for person {example_person}.")
#     gesture_indices.append(matches[0])

# plot_feature_comparison(
#     gesture_indices,
#     gestures,
#     f"Four gestures from training person {example_person} — run {analysis_run}",
#     "01_four_gestures.png",
# )
# data.iloc[gesture_indices].to_csv(analysis_output / "01_selected_recordings.csv", index=False)


# # 2. Compare one gesture across every training person in this run.
# person_indices = []

# for person in analysis_train_people:
#     matches = np.flatnonzero(
#         analysis_mask
#         & (data["person"].to_numpy() == person)
#         & (data["gesture"].to_numpy() == example_gesture)
#     )
#     if len(matches) == 0:
#         raise ValueError(f"No {example_gesture} recordings found for person {person}.")
#     person_indices.append(matches[0])

# plot_feature_comparison(
#     person_indices,
#     [f"Person {person}" for person in analysis_train_people],
#     f"{example_gesture.capitalize()} across training people — run {analysis_run}",
#     "02_training_people.png",
# )
# data.iloc[person_indices].to_csv(analysis_output / "02_selected_recordings.csv", index=False)


# # 3. Compare feature distributions using ALL training recordings.
# # Show two existing features from each family; the full table still has 94 features.
# selected_features = {
#     "envelope_positive_16": "Positive envelope near the gesture midpoint",
#     "envelope_negative_16": "Negative envelope near the gesture midpoint",
#     "centroid_mean": "Mean Doppler centroid",
#     "bandwidth_mean": "Mean RMS Doppler bandwidth",
#     "chebyshev_d0_c0": "Chebyshev moment: Doppler order 0, cadence order 0",
#     "chebyshev_d0_c1": "Chebyshev moment: Doppler order 0, cadence order 1",
# }

# fig, axes = plt.subplots(3, 2, figsize=(12, 11), layout="constrained")

# for ax, (feature_name, title) in zip(axes.ravel(), selected_features.items()):
#     groups = []
#     labels = []

#     for gesture in gestures:
#         mask = analysis_mask & (data["gesture"].to_numpy() == gesture)
#         values = features.loc[mask, feature_name].to_numpy()
#         groups.append(values)
#         labels.append(f"{gesture}\n(n={len(values)})")

#     ax.boxplot(groups, showfliers=True)
#     ax.set_xticks(np.arange(1, len(gestures) + 1))
#     ax.set_xticklabels(labels)
#     ax.set_title(title, fontsize=10)
#     ax.set_ylabel("Moment coefficient" if feature_name.startswith("chebyshev")
#                   else "Relative Doppler units")
#     ax.grid(axis="y", alpha=0.2)

# fig.suptitle(
#     f"Feature distributions across all {analysis_mask.sum()} training recordings — run {analysis_run}",
#     fontsize=14,
# )
# fig.canvas.draw()
# fig.savefig(analysis_output / "03_feature_distributions.png", dpi=150, bbox_inches="tight")
# plt.show()
# plt.close(fig)

# # Save descriptive statistics for all 94 features, not just the six shown above.
# summary_rows = []

# for gesture in gestures:
#     mask = analysis_mask & (data["gesture"].to_numpy() == gesture)
#     for feature_name in features.columns:
#         values = features.loc[mask, feature_name]
#         summary_rows.append({
#             "gesture": gesture, "feature": feature_name,
#             "count": len(values), "mean": values.mean(), "std": values.std(),
#             "median": values.median(), "minimum": values.min(), "maximum": values.max(),
#         })

# pd.DataFrame(summary_rows).to_csv(analysis_output / "feature_statistics.csv", index=False)
# pd.DataFrame({"moment_index": np.arange(len(moment_columns)), "feature": moment_columns}).to_csv(
#     analysis_output / "moment_names.csv", index=False
# )

# print("Analysis run:", analysis_run)
# print("Training people analysed:", analysis_train_people)
# print("Training recordings analysed:", analysis_mask.sum())
# print("Analysis figures and tables saved to:", analysis_output)
