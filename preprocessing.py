from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# creating the output folder
root = Path(r"C:\Users\SID-DRW\Downloads\WICOS - Copy")
output = root / "Project" / "Project_Output"
output.mkdir(parents=True, exist_ok=True)

files = sorted(root.rglob("Person*g*s*.csv"))
print(f"Found {len(files)} recordings")


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


# extract features once
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

    y_train = y[train_mask]
    y_validation = y[validation_mask]
    y_test = y[test_mask]

    print(f"\nRun {run}")
    print("Training people:", train_people)
    print("Validation person:", validation_person)
    print("Test person:", test_person)
    print("Training feature shape:", X_features[train_mask].shape)

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
