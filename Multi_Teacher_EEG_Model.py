
import os
import csv
import math
import random
import warnings

import numpy as np
import scipy.io as sio
import tensorflow as tf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from keras.layers import (
    Input, Dense, Dropout, BatchNormalization, LayerNormalization,
    Conv1D, SeparableConv1D, SpatialDropout1D,
    MultiHeadAttention, Add, Multiply, Concatenate,
    Lambda, Softmax, Activation, Flatten
)
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import Callback, ReduceLROnPlateau
from keras.losses import CategoricalCrossentropy
from keras.regularizers import l2

from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report,
    matthews_corrcoef, cohen_kappa_score,
    roc_auc_score, roc_curve, auc
)

warnings.filterwarnings("ignore", category=FutureWarning)

CosineDecay = tf.keras.optimizers.schedules.CosineDecay

SEED = 42

DATASET_PATH = r"F:/EEG Emotion Detection/SEED-IV Dataset/eeg_feature_smooth"
RESULTS_DIR = r"F:/EEG Emotion Detection Jupyter/Results/Model 13.8 Outputs"
FIGURES_DIR = os.path.join(RESULTS_DIR, "Figures")

SPLIT_PROTOCOL = "window"
FEATURE_MODE = "multi_eeg"

NUM_CLASSES = 4
CLASS_NAMES = ["Neutral", "Sad", "Fear", "Happy"]

TEST_SIZE = 0.20
VAL_SIZE_FROM_TRAINVAL = 0.15

TEACHER_EPOCHS = 50
STUDENT_EPOCHS = 40

TEACHER_BATCH_SIZE = 256
STUDENT_BATCH_SIZE = 256

LABEL_SMOOTHING = 0.02
WEIGHT_DECAY = 4e-5

TEACHER_LR = 2.0e-4
STUDENT_LR = 2.5e-4

TEMPERATURE = 4.0
KD_MIN_TEACHER_VAL_F1 = 0.70
ALPHA_HARD_STRONG_TEACHER = 0.65
ALPHA_HARD_WEAK_TEACHER = 0.88

AUG_PROB = 0.30


def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


seed_everything(SEED)

try:
    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) > 0:
        tf.config.set_visible_devices(gpus[0], "GPU")
        tf.config.experimental.set_memory_growth(gpus[0], True)
except Exception:
    pass


session1_label = [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3]
session2_label = [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1]
session3_label = [1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0]

session_labels = [session1_label, session2_label, session3_label]


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def save_text(path, text):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def write_dict_csv(path, rows, fieldnames):
    ensure_dir(os.path.dirname(path))

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def infer_subject_id(filename):
    base = os.path.splitext(filename)[0]
    return base.split("_")[0]


def get_feature_prefixes():
    if FEATURE_MODE == "de_only":
        return ["de_LDS"]

    if FEATURE_MODE == "multi_eeg":
        return ["de_LDS", "de_movingAve", "psd_LDS", "psd_movingAve"]

    raise ValueError("FEATURE_MODE must be 'de_only' or 'multi_eeg'.")


def load_one_trial_de_only(mat, trial_num):
    key = f"de_LDS{trial_num}"

    if key not in mat:
        raise KeyError(f"Missing key: {key}")

    arr = np.asarray(mat[key], dtype=np.float32)

    if arr.ndim != 3:
        raise ValueError(f"Expected 3D EEG feature array for {key}, got {arr.shape}")

    if arr.shape[0] != 62 or arr.shape[2] != 5:
        raise ValueError(f"Expected shape 62 x T x 5 for {key}, got {arr.shape}")

    return arr.transpose(1, 0, 2).astype(np.float32)


def load_one_trial_multifeature(mat, trial_num):
    """
    Returns one trial as:
        time_windows x 62 channels x feature_dim

    feature_dim:
        de_only   = 5
        multi_eeg = 20
    """
    feature_arrays = []

    for prefix in get_feature_prefixes():
        key = f"{prefix}{trial_num}"

        if key not in mat:
            if FEATURE_MODE == "multi_eeg":
                return load_one_trial_de_only(mat, trial_num)
            raise KeyError(f"Missing key: {key}")

        arr = np.asarray(mat[key], dtype=np.float32)

        if arr.ndim != 3:
            raise ValueError(f"Expected 3D EEG feature array for {key}, got {arr.shape}")

        if arr.shape[0] != 62 or arr.shape[2] != 5:
            raise ValueError(f"Expected shape 62 x T x 5 for {key}, got {arr.shape}")

        if prefix.startswith("psd"):
            arr = np.log1p(np.maximum(arr, 0.0)).astype(np.float32)

        feature_arrays.append(arr)

    min_t = min(a.shape[1] for a in feature_arrays)
    feature_arrays = [a[:, :min_t, :] for a in feature_arrays]

    fused = np.concatenate(feature_arrays, axis=2)
    fused = fused.transpose(1, 0, 2).astype(np.float32)

    return fused


def load_seediv_trials():
    X_trials = []
    y_trials = []
    groups = []
    meta = []

    sessions = sorted([
        d for d in os.listdir(DATASET_PATH)
        if os.path.isdir(os.path.join(DATASET_PATH, d))
    ])

    if len(sessions) != 3:
        raise ValueError(f"Expected 3 session folders, found {len(sessions)}: {sessions}")

    for session_index, session in enumerate(sessions):
        session_path = os.path.join(DATASET_PATH, session)
        labels = session_labels[session_index]

        mat_files = sorted([
            f for f in os.listdir(session_path)
            if f.endswith(".mat")
        ])

        print(f"Session {session}: {len(mat_files)} EEG .mat files")

        for file in mat_files:
            path = os.path.join(session_path, file)
            mat = sio.loadmat(path)

            subject_id = infer_subject_id(file)

            de_keys = [
                k for k in mat.keys()
                if k.startswith("de_LDS") and k.replace("de_LDS", "").isdigit()
            ]

            de_keys = sorted(de_keys, key=lambda k: int(k.replace("de_LDS", "")))

            if len(de_keys) != 24:
                print(f"Skipping {session}/{file}: expected 24 de_LDS keys, found {len(de_keys)}")
                continue

            for trial_num in range(1, 25):
                trial_idx = trial_num - 1
                trial = load_one_trial_multifeature(mat, trial_num)

                X_trials.append(trial)
                y_trials.append(labels[trial_idx])
                groups.append(subject_id)
                meta.append((session, file, trial_idx, trial.shape[0]))

    if len(X_trials) == 0:
        raise RuntimeError("No trials loaded. Check DATASET_PATH.")

    return (
        X_trials,
        np.asarray(y_trials, dtype=np.int64),
        np.asarray(groups),
        meta
    )


def trials_to_windows(X_trials, y_trials, trial_indices):
    X = []
    y = []
    trial_ids = []
    time_ids = []

    for trial_id in trial_indices:
        trial = X_trials[trial_id]
        label = int(y_trials[trial_id])

        for t in range(trial.shape[0]):
            X.append(trial[t])
            y.append(label)
            trial_ids.append(trial_id)
            time_ids.append(t)

    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.int64),
        np.asarray(trial_ids, dtype=np.int64),
        np.asarray(time_ids, dtype=np.int64)
    )


def split_windows_directly(X_all, y_all, trial_ids_all, time_ids_all):
    indices = np.arange(len(y_all))

    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=y_all
    )

    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=VAL_SIZE_FROM_TRAINVAL,
        random_state=SEED + 1,
        stratify=y_all[train_val_idx]
    )

    return (
        X_all[train_idx],
        y_all[train_idx],
        trial_ids_all[train_idx],
        time_ids_all[train_idx],

        X_all[val_idx],
        y_all[val_idx],
        trial_ids_all[val_idx],
        time_ids_all[val_idx],

        X_all[test_idx],
        y_all[test_idx],
        trial_ids_all[test_idx],
        time_ids_all[test_idx],
    )


def normalize_windows(X_train, X_val, X_test):
    original_shape = X_train.shape[1:]
    flat_dim = int(np.prod(original_shape))

    scaler = StandardScaler()
    scaler.fit(X_train.reshape(len(X_train), flat_dim))

    def transform(X):
        X_flat = X.reshape(len(X), flat_dim)
        X_scaled = scaler.transform(X_flat).astype(np.float32)
        return X_scaled.reshape((-1,) + original_shape)

    return transform(X_train), transform(X_val), transform(X_test), scaler


def make_sample_weights(y):
    counts = np.bincount(y, minlength=NUM_CLASSES).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = np.max(counts) / counts
    return weights[y].astype(np.float32)


def specificity_from_confusion(cm):
    total = np.sum(cm)
    rows = []
    specs = []

    for i, class_name in enumerate(CLASS_NAMES):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        tn = total - tp - fp - fn

        specificity = tn / (tn + fp + 1e-12)
        specs.append(specificity)

        rows.append({
            "class": class_name,
            "specificity": float(specificity),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp)
        })

    return rows, float(np.mean(specs))


def safe_auc_scores(y_true, probs):
    y_bin = label_binarize(y_true, classes=list(range(NUM_CLASSES)))

    out = {
        "auc_macro_ovr": np.nan,
        "auc_weighted_ovr": np.nan,
        "auc_micro": np.nan
    }

    try:
        out["auc_macro_ovr"] = float(
            roc_auc_score(y_bin, probs, average="macro", multi_class="ovr")
        )
    except Exception:
        pass

    try:
        out["auc_weighted_ovr"] = float(
            roc_auc_score(y_bin, probs, average="weighted", multi_class="ovr")
        )
    except Exception:
        pass

    try:
        out["auc_micro"] = float(
            roc_auc_score(y_bin.ravel(), probs.ravel())
        )
    except Exception:
        pass

    class_auc = {}

    for i, class_name in enumerate(CLASS_NAMES):
        try:
            if len(np.unique(y_bin[:, i])) == 2:
                class_auc[class_name] = float(roc_auc_score(y_bin[:, i], probs[:, i]))
            else:
                class_auc[class_name] = np.nan
        except Exception:
            class_auc[class_name] = np.nan

    return out, class_auc


def plot_confusion_matrix(cm, title, path):
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()

    ticks = np.arange(NUM_CLASSES)
    plt.xticks(ticks, CLASS_NAMES, rotation=45, ha="right")
    plt.yticks(ticks, CLASS_NAMES)

    threshold = cm.max() / 2.0 if cm.max() > 0 else 0

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j]
            plt.text(
                j, i, str(value),
                ha="center",
                va="center",
                color="white" if value > threshold else "black"
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_roc_curves(y_true, probs, title, path):
    y_bin = label_binarize(y_true, classes=list(range(NUM_CLASSES)))

    plt.figure(figsize=(8, 7))

    for i, class_name in enumerate(CLASS_NAMES):
        try:
            if len(np.unique(y_bin[:, i])) < 2:
                continue

            fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
            auc_value = auc(fpr, tpr)

            plt.plot(fpr, tpr, label=f"{class_name} AUC={auc_value:.4f}")

        except Exception:
            continue

    try:
        fpr_micro, tpr_micro, _ = roc_curve(y_bin.ravel(), probs.ravel())
        auc_micro = auc(fpr_micro, tpr_micro)

        plt.plot(
            fpr_micro,
            tpr_micro,
            linestyle="--",
            label=f"Micro-average AUC={auc_micro:.4f}"
        )

    except Exception:
        pass

    plt.plot([0, 1], [0, 1], linestyle=":", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_final_accuracy_bar(metrics_rows, path):
    split_names = ["train", "val", "test"]
    values = []

    for split in split_names:
        for row in metrics_rows:
            if row["split"] == split:
                values.append(row["accuracy"])
                break

    plt.figure(figsize=(7, 5))
    plt.bar([s.upper() for s in split_names], values)
    plt.ylim(0, 1.05)
    plt.ylabel("Accuracy")
    plt.title("Train / Validation / Test Accuracy")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def evaluate_and_save(split_name, y_true, probs):
    y_pred = np.argmax(probs, axis=1)

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES))
    )

    accuracy = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted"))
    mcc = float(matthews_corrcoef(y_true, y_pred))
    kappa = float(cohen_kappa_score(y_true, y_pred))
    specificity_rows, specificity_macro = specificity_from_confusion(cm)
    auc_scores, class_auc = safe_auc_scores(y_true, probs)

    report_text = classification_report(
        y_true,
        y_pred,
        labels=list(range(NUM_CLASSES)),
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0
    )

    print()
    print("=" * 72)
    print(f"{split_name.upper()} CLASSIFICATION REPORT")
    print("=" * 72)
    print(report_text)
    print(f"{split_name.upper()} Accuracy: {accuracy:.4f}")
    print(f"{split_name.upper()} Macro-F1: {macro_f1:.4f}")
    print(f"{split_name.upper()} MCC: {mcc:.4f}")
    print(f"{split_name.upper()} Kappa: {kappa:.4f}")
    print(f"{split_name.upper()} Specificity: {specificity_macro:.4f}")
    print(f"{split_name.upper()} AUC Macro OVR: {auc_scores['auc_macro_ovr']:.4f}")

    save_text(
        os.path.join(FIGURES_DIR, f"classification_report_{split_name}.txt"),
        report_text
    )

    plot_confusion_matrix(
        cm,
        f"{split_name.upper()} Confusion Matrix",
        os.path.join(FIGURES_DIR, f"confusion_matrix_{split_name}.png")
    )

    plot_roc_curves(
        y_true,
        probs,
        f"{split_name.upper()} ROC Curve",
        os.path.join(FIGURES_DIR, f"roc_curve_{split_name}.png")
    )

    specificity_path = os.path.join(FIGURES_DIR, f"specificity_{split_name}.csv")
    write_dict_csv(
        specificity_path,
        specificity_rows,
        fieldnames=["class", "specificity", "tn", "fp", "fn", "tp"]
    )

    row = {
        "split": split_name,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "mcc": mcc,
        "kappa": kappa,
        "specificity_macro": specificity_macro,
        "auc_macro_ovr": auc_scores["auc_macro_ovr"],
        "auc_weighted_ovr": auc_scores["auc_weighted_ovr"],
        "auc_micro": auc_scores["auc_micro"],
        "auc_Neutral": class_auc["Neutral"],
        "auc_Sad": class_auc["Sad"],
        "auc_Fear": class_auc["Fear"],
        "auc_Happy": class_auc["Happy"]
    }

    return row


def feature_attention(inputs, feature_dim, name):
    f = Lambda(
        lambda t: tf.reduce_mean(t, axis=1),
        name=name + "_feature_stat"
    )(inputs)

    f = Dense(
        max(8, feature_dim * 2),
        activation="gelu",
        kernel_regularizer=l2(WEIGHT_DECAY),
        name=name + "_feature_dense1"
    )(f)

    f = Dense(
        feature_dim,
        activation="sigmoid",
        kernel_regularizer=l2(WEIGHT_DECAY),
        name=name + "_feature_dense2"
    )(f)

    f = Lambda(
        lambda t: tf.expand_dims(t, axis=1),
        name=name + "_feature_expand"
    )(f)

    x = Multiply(name=name + "_feature_scale")([inputs, f])

    c = Lambda(
        lambda t: tf.reduce_mean(t, axis=2),
        name=name + "_channel_stat"
    )(x)

    c = Dense(
        16,
        activation="gelu",
        kernel_regularizer=l2(WEIGHT_DECAY),
        name=name + "_channel_dense1"
    )(c)

    channel_dim = int(inputs.shape[1]) if inputs.shape[1] is not None else 62

    c = Dense(
        channel_dim,
        activation="sigmoid",
        kernel_regularizer=l2(WEIGHT_DECAY),
        name=name + "_channel_dense2"
    )(c)

    c = Lambda(
        lambda t: tf.expand_dims(t, axis=2),
        name=name + "_channel_expand"
    )(c)

    x = Multiply(name=name + "_channel_scale")([x, c])

    return x


def attention_pool(x, name):
    score = Dense(1, use_bias=False, name=name + "_score")(x)

    score = Lambda(
        lambda t: tf.squeeze(t, axis=-1),
        name=name + "_squeeze"
    )(score)

    weights = Softmax(axis=1, name=name + "_softmax")(score)

    weights = Lambda(
        lambda t: tf.expand_dims(t, axis=-1),
        name=name + "_expand"
    )(weights)

    return Lambda(
        lambda args: tf.reduce_sum(args[0] * args[1], axis=1),
        name=name
    )([x, weights])


def multi_pool(x, name):
    att = attention_pool(x, name + "_att")

    mean = Lambda(
        lambda t: tf.reduce_mean(t, axis=1),
        name=name + "_mean"
    )(x)

    mx = Lambda(
        lambda t: tf.reduce_max(t, axis=1),
        name=name + "_max"
    )(x)

    return Concatenate(name=name + "_concat")([att, mean, mx])


def multiscale_spatial_block(x, filters, dropout, name):
    shortcut = x

    if int(x.shape[-1]) != filters:
        shortcut = Conv1D(
            filters,
            1,
            padding="same",
            use_bias=False,
            kernel_regularizer=l2(WEIGHT_DECAY),
            name=name + "_shortcut"
        )(shortcut)

        shortcut = BatchNormalization(name=name + "_shortcut_bn")(shortcut)

    branch_filters = max(16, filters // 4)

    b1 = SeparableConv1D(
        branch_filters,
        1,
        padding="same",
        use_bias=False,
        depthwise_regularizer=l2(WEIGHT_DECAY),
        pointwise_regularizer=l2(WEIGHT_DECAY),
        name=name + "_k1"
    )(x)

    b3 = SeparableConv1D(
        branch_filters,
        3,
        padding="same",
        use_bias=False,
        depthwise_regularizer=l2(WEIGHT_DECAY),
        pointwise_regularizer=l2(WEIGHT_DECAY),
        name=name + "_k3"
    )(x)

    b5 = SeparableConv1D(
        branch_filters,
        5,
        padding="same",
        use_bias=False,
        depthwise_regularizer=l2(WEIGHT_DECAY),
        pointwise_regularizer=l2(WEIGHT_DECAY),
        name=name + "_k5"
    )(x)

    b9 = SeparableConv1D(
        branch_filters,
        9,
        padding="same",
        use_bias=False,
        depthwise_regularizer=l2(WEIGHT_DECAY),
        pointwise_regularizer=l2(WEIGHT_DECAY),
        name=name + "_k9"
    )(x)

    y = Concatenate(name=name + "_concat")([b1, b3, b5, b9])

    y = BatchNormalization(name=name + "_bn1")(y)
    y = Activation("gelu", name=name + "_act1")(y)

    y = Conv1D(
        filters,
        1,
        padding="same",
        use_bias=False,
        kernel_regularizer=l2(WEIGHT_DECAY),
        name=name + "_project"
    )(y)

    y = BatchNormalization(name=name + "_bn2")(y)

    gate = Dense(
        filters,
        activation="sigmoid",
        kernel_regularizer=l2(WEIGHT_DECAY),
        name=name + "_gate"
    )(y)

    y = Multiply(name=name + "_gated")([y, gate])
    y = Add(name=name + "_add")([shortcut, y])
    y = Activation("gelu", name=name + "_out")(y)
    y = SpatialDropout1D(dropout, name=name + "_sdrop")(y)

    return y


def transformer_encoder(x, d_model, heads, ff_mult, dropout, name):
    attn = MultiHeadAttention(
        num_heads=heads,
        key_dim=max(1, d_model // heads),
        dropout=dropout,
        name=name + "_mha"
    )(x, x)

    gate = Dense(
        d_model,
        activation="sigmoid",
        kernel_regularizer=l2(WEIGHT_DECAY),
        name=name + "_attn_gate"
    )(x)

    attn = Multiply(name=name + "_attn_gated")([attn, gate])
    attn = Dropout(dropout, name=name + "_attn_drop")(attn)

    x = Add(name=name + "_attn_add")([x, attn])
    x = LayerNormalization(epsilon=1e-6, name=name + "_ln1")(x)

    ff = Dense(
        d_model * ff_mult,
        activation="gelu",
        kernel_regularizer=l2(WEIGHT_DECAY),
        name=name + "_ff1"
    )(x)

    ff = Dropout(dropout, name=name + "_ff_drop1")(ff)

    ff = Dense(
        d_model,
        kernel_regularizer=l2(WEIGHT_DECAY),
        name=name + "_ff2"
    )(ff)

    ff_gate = Dense(
        d_model,
        activation="sigmoid",
        kernel_regularizer=l2(WEIGHT_DECAY),
        name=name + "_ff_gate"
    )(x)

    ff = Multiply(name=name + "_ff_gated")([ff, ff_gate])
    ff = Dropout(dropout, name=name + "_ff_drop2")(ff)

    x = Add(name=name + "_ff_add")([x, ff])
    x = LayerNormalization(epsilon=1e-6, name=name + "_ln2")(x)

    return x


def separable_student_block(x, filters, kernel_size, dropout, name):
    shortcut = x

    if int(x.shape[-1]) != filters:
        shortcut = Conv1D(
            filters,
            1,
            padding="same",
            use_bias=False,
            kernel_regularizer=l2(WEIGHT_DECAY),
            name=name + "_shortcut"
        )(shortcut)

        shortcut = BatchNormalization(name=name + "_shortcut_bn")(shortcut)

    y = SeparableConv1D(
        filters,
        kernel_size,
        padding="same",
        use_bias=False,
        depthwise_regularizer=l2(WEIGHT_DECAY),
        pointwise_regularizer=l2(WEIGHT_DECAY),
        name=name + "_sepconv"
    )(x)

    y = BatchNormalization(name=name + "_bn1")(y)
    y = Activation("gelu", name=name + "_act1")(y)
    y = SpatialDropout1D(dropout, name=name + "_sdrop1")(y)

    y = Conv1D(
        filters,
        1,
        padding="same",
        use_bias=False,
        kernel_regularizer=l2(WEIGHT_DECAY),
        name=name + "_project"
    )(y)

    y = BatchNormalization(name=name + "_bn2")(y)

    y = Add(name=name + "_add")([shortcut, y])
    y = Activation("gelu", name=name + "_out")(y)

    return y



def build_teacher(input_shape):
    inputs = Input(shape=input_shape, name="eeg_window_input")

    x = Conv1D(
        128,
        1,
        padding="same",
        use_bias=False,
        kernel_regularizer=l2(WEIGHT_DECAY),
        name="teacher1_cnn_first_stem"
    )(inputs)

    x = BatchNormalization(name="teacher1_cnn_first_stem_bn")(x)
    x = Activation("gelu", name="teacher1_cnn_first_stem_act")(x)

    x_att = feature_attention(x, 128, name="teacher1_postcnn_fa")

    x = multiscale_spatial_block(x_att, 128, 0.10, "teacher1_ms1")
    x = multiscale_spatial_block(x, 160, 0.12, "teacher1_ms2")
    x = multiscale_spatial_block(x, 160, 0.12, "teacher1_ms3")

    for i in range(3):
        x = transformer_encoder(
            x,
            d_model=160,
            heads=4,
            ff_mult=3,
            dropout=0.12,
            name=f"teacher1_transformer_{i + 1}"
        )

    pooled = multi_pool(x, name="teacher1_pool")

    flat = Flatten(name="teacher1_flatten_attended_cnn")(x_att)

    flat = Dense(
        256,
        activation="gelu",
        kernel_regularizer=l2(WEIGHT_DECAY),
        name="teacher1_flat_dense1"
    )(flat)

    flat = Dropout(0.25, name="teacher1_flat_drop1")(flat)

    z = Concatenate(name="teacher1_fusion")([pooled, flat])

    z = Dense(
        384,
        activation="gelu",
        kernel_regularizer=l2(WEIGHT_DECAY),
        name="teacher1_dense1"
    )(z)

    z = Dropout(0.35, name="teacher1_drop1")(z)

    z = Dense(
        192,
        activation="gelu",
        kernel_regularizer=l2(WEIGHT_DECAY),
        name="teacher1_dense2"
    )(z)

    z = Dropout(0.25, name="teacher1_drop2")(z)

    logits = Dense(NUM_CLASSES, name="logits")(z)

    return Model(inputs, logits, name="M13_8_Teacher1_CNNFirst_Attention_CNN_Transformer")



def build_teacher2(input_shape):
    inputs = Input(shape=input_shape, name="eeg_window_input")

    x = LayerNormalization(epsilon=1e-6, name="teacher2_input_ln")(inputs)

    x = Dense(
        96,
        activation="gelu",
        kernel_regularizer=l2(WEIGHT_DECAY),
        name="teacher2_token_projection"
    )(x)

    x = Dropout(0.08, name="teacher2_projection_drop")(x)

    for i in range(2):
        x = transformer_encoder(
            x,
            d_model=96,
            heads=4,
            ff_mult=2,
            dropout=0.10,
            name=f"teacher2_light_transformer_{i + 1}"
        )

    pooled = multi_pool(x, name="teacher2_pool")

    z = Dense(
        192,
        activation="gelu",
        kernel_regularizer=l2(WEIGHT_DECAY),
        name="teacher2_dense1"
    )(pooled)

    z = Dropout(0.25, name="teacher2_drop1")(z)

    z = Dense(
        96,
        activation="gelu",
        kernel_regularizer=l2(WEIGHT_DECAY),
        name="teacher2_dense2"
    )(z)

    z = Dropout(0.18, name="teacher2_drop2")(z)

    logits = Dense(NUM_CLASSES, name="logits")(z)

    return Model(inputs, logits, name="M13_8_Teacher2_Pure_LightTransformer")



def build_student(input_shape):
    inputs = Input(shape=input_shape, name="eeg_window_input")

    x = Conv1D(
        64,
        1,
        padding="same",
        use_bias=False,
        kernel_regularizer=l2(WEIGHT_DECAY),
        name="student_stem"
    )(inputs)

    x = BatchNormalization(name="student_stem_bn")(x)
    x = Activation("gelu", name="student_stem_act")(x)

    x = separable_student_block(x, 64, 5, 0.10, "student_sep1")
    x = separable_student_block(x, 96, 5, 0.12, "student_sep2")
    x = separable_student_block(x, 96, 3, 0.12, "student_sep3")

    x = Conv1D(
        96,
        1,
        padding="same",
        use_bias=False,
        kernel_regularizer=l2(WEIGHT_DECAY),
        name="student_projection"
    )(x)

    x = BatchNormalization(name="student_projection_bn")(x)
    x = Activation("gelu", name="student_projection_act")(x)

    lstm_cell = tf.keras.layers.LSTMCell(
        96,
        dropout=0.10,
        recurrent_dropout=0.05,
        kernel_regularizer=l2(WEIGHT_DECAY),
        recurrent_regularizer=l2(WEIGHT_DECAY),
        name="student_lstm_cell_1"
    )

    lstm_out = tf.keras.layers.RNN(
        lstm_cell,
        return_sequences=True,
        name="student_lstm_1"
    )(x)

    lstm_out = Dropout(0.10, name="student_lstm_drop")(lstm_out)

    x = Add(name="student_lstm_residual_add")([x, lstm_out])
    x = LayerNormalization(epsilon=1e-6, name="student_lstm_ln")(x)

    pooled = multi_pool(x, name="student_pool")

    flat = Flatten(name="student_flatten_input")(inputs)

    flat = Dense(
        128,
        activation="gelu",
        kernel_regularizer=l2(WEIGHT_DECAY),
        name="student_flat_dense1"
    )(flat)

    flat = Dropout(0.20, name="student_flat_drop1")(flat)

    z = Concatenate(name="student_fusion")([pooled, flat])

    z = Dense(
        192,
        activation="gelu",
        kernel_regularizer=l2(WEIGHT_DECAY),
        name="student_dense1"
    )(z)

    z = Dropout(0.28, name="student_drop1")(z)

    z = Dense(
        96,
        activation="gelu",
        kernel_regularizer=l2(WEIGHT_DECAY),
        name="student_dense2"
    )(z)

    z = Dropout(0.20, name="student_drop2")(z)

    logits = Dense(NUM_CLASSES, name="logits")(z)

    return Model(inputs, logits, name="M12_Window_Student_CNN_LSTM_KD")


def make_optimizer(lr):
    return Adam(learning_rate=lr, clipnorm=1.0)


def save_model_architecture(path, models):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        for model in models:
            f.write("=" * 72 + "\n")
            f.write(model.name + "\n")
            f.write(f"Parameters: {model.count_params()}\n")
            f.write("=" * 72 + "\n")
            model.summary(print_fn=lambda line: f.write(line + "\n"))
            f.write("\n\n")


class ValMacroF1BestWeights(Callback):
    def __init__(self, X_val, y_val, batch_size=512, name="model"):
        super().__init__()

        self.X_val = X_val
        self.y_val = y_val
        self.batch_size = batch_size
        self.name = name

        self.best_f1 = -np.inf
        self.best_acc = -np.inf
        self.best_weights = None

    def on_epoch_end(self, epoch, logs=None):
        logits = self.model.predict(
            self.X_val,
            batch_size=self.batch_size,
            verbose=0
        )

        pred = np.argmax(logits, axis=1)

        val_acc = float(np.mean(pred == self.y_val))
        val_f1 = float(f1_score(self.y_val, pred, average="macro"))

        print(f"{self.name} validation accuracy: {val_acc:.4f} | validation macro-F1: {val_f1:.4f}")

        if val_f1 > self.best_f1 + 1e-5:
            self.best_f1 = val_f1
            self.best_acc = val_acc
            self.best_weights = self.model.get_weights()

    def on_train_end(self, logs=None):
        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)
            print(f"{self.name} restored best weights. Best val acc={self.best_acc:.4f}, best val F1={self.best_f1:.4f}")


def model_probs(model, X, batch_size=512, temperature=1.0):
    logits = model.predict(X, batch_size=batch_size, verbose=0)
    return tf.nn.softmax(logits / temperature, axis=1).numpy()


def augment_windows(batch_x, prob=AUG_PROB):
    x = batch_x.copy()
    bs, channels, features = x.shape

    for i in range(bs):
        if random.random() < prob:
            num_ch = random.randint(1, 5)
            ch_idx = np.random.choice(channels, size=num_ch, replace=False)
            x[i, ch_idx, :] = 0.0

        if random.random() < prob:
            if features % 5 == 0:
                band = random.randint(0, 4)
                x[i, :, band::5] = 0.0
            else:
                feat = random.randint(0, features - 1)
                x[i, :, feat] = 0.0

        if random.random() < prob:
            noise = np.random.normal(
                0.0,
                random.uniform(0.002, 0.010),
                size=x[i].shape
            ).astype(np.float32)

            x[i] += noise

        if random.random() < prob:
            x[i] *= random.uniform(0.96, 1.04)

    return x.astype(np.float32)


def train_teacher_model(model, X_train, y_train, X_val, y_val, teacher_name="Teacher"):
    print()
    print("=" * 72)
    print(f"TRAINING {teacher_name.upper()}")
    print("=" * 72)

    model.compile(
        optimizer=make_optimizer(TEACHER_LR),
        loss=CategoricalCrossentropy(
            from_logits=True,
            label_smoothing=LABEL_SMOOTHING
        ),
        metrics=["accuracy"]
    )

    sample_weight = make_sample_weights(y_train)

    callbacks = [
        ValMacroF1BestWeights(
            X_val,
            y_val,
            batch_size=512,
            name=teacher_name
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.5,
            patience=3,
            min_lr=5e-7,
            verbose=1
        )
    ]

    model.fit(
        X_train,
        tf.one_hot(y_train, NUM_CLASSES).numpy(),
        sample_weight=sample_weight,
        validation_data=(
            X_val,
            tf.one_hot(y_val, NUM_CLASSES).numpy()
        ),
        epochs=TEACHER_EPOCHS,
        batch_size=TEACHER_BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    model.trainable = False

    return model, callbacks[0].best_f1


def train_student_kd(student, teacher1, teacher2, X_train, y_train, X_val, y_val, teacher1_val_f1, teacher2_val_f1):
    print()
    print("=" * 72)
    print("TRAINING STUDENT WITH TWO-TEACHER KNOWLEDGE DISTILLATION")
    print("=" * 72)

    print("Computing Teacher 1 soft labels...")

    teacher1_soft_train = model_probs(
        teacher1,
        X_train,
        batch_size=512,
        temperature=TEMPERATURE
    )

    print("Computing Teacher 2 soft labels...")

    teacher2_soft_train = model_probs(
        teacher2,
        X_train,
        batch_size=512,
        temperature=TEMPERATURE
    )

    f1_sum = teacher1_val_f1 + teacher2_val_f1 + 1e-8

    teacher1_weight = teacher1_val_f1 / f1_sum
    teacher2_weight = teacher2_val_f1 / f1_sum

    teacher_soft_train = (
        teacher1_weight * teacher1_soft_train +
        teacher2_weight * teacher2_soft_train
    ).astype(np.float32)

    print(f"Teacher 1 soft-label weight: {teacher1_weight:.4f}")
    print(f"Teacher 2 soft-label weight: {teacher2_weight:.4f}")

    best_teacher_val_f1 = max(teacher1_val_f1, teacher2_val_f1)

    if best_teacher_val_f1 >= KD_MIN_TEACHER_VAL_F1:
        alpha_hard = ALPHA_HARD_STRONG_TEACHER
    else:
        alpha_hard = ALPHA_HARD_WEAK_TEACHER

    print(f"KD hard-label weight alpha: {alpha_hard:.2f}")

    steps_per_epoch = math.ceil(len(X_train) / STUDENT_BATCH_SIZE)

    lr_schedule = CosineDecay(
        initial_learning_rate=STUDENT_LR,
        decay_steps=STUDENT_EPOCHS * steps_per_epoch,
        alpha=0.04
    )

    optimizer = make_optimizer(lr_schedule)
    sample_weight = make_sample_weights(y_train)

    eps = 1e-7
    best_f1 = -np.inf
    best_acc = -np.inf
    best_weights = None

    for epoch in range(STUDENT_EPOCHS):
        indices = np.arange(len(X_train))
        np.random.shuffle(indices)

        losses = []

        for step in range(steps_per_epoch):
            start = step * STUDENT_BATCH_SIZE
            end = min(start + STUDENT_BATCH_SIZE, len(indices))

            batch_idx = indices[start:end]

            bx = X_train[batch_idx]
            by = y_train[batch_idx].astype(np.int64)
            bt = teacher_soft_train[batch_idx].astype(np.float32)
            bw = sample_weight[batch_idx].astype(np.float32)

            bx = augment_windows(bx, prob=AUG_PROB)
            by_onehot = tf.one_hot(by, NUM_CLASSES)

            with tf.GradientTape() as tape:
                logits = student(bx, training=True)

                hard_loss_per_sample = tf.keras.losses.categorical_crossentropy(
                    by_onehot,
                    logits,
                    from_logits=True,
                    label_smoothing=LABEL_SMOOTHING
                )

                hard_loss = tf.reduce_mean(hard_loss_per_sample * bw)

                student_soft = tf.nn.softmax(logits / TEMPERATURE, axis=1)
                teacher_soft = tf.convert_to_tensor(bt, dtype=tf.float32)

                kl = tf.reduce_sum(
                    teacher_soft * (
                        tf.math.log(teacher_soft + eps) -
                        tf.math.log(student_soft + eps)
                    ),
                    axis=1
                )

                soft_loss = tf.reduce_mean(kl * bw) * (TEMPERATURE ** 2)

                loss = alpha_hard * hard_loss + (1.0 - alpha_hard) * soft_loss

                if student.losses:
                    loss += tf.add_n(student.losses)

            grads = tape.gradient(loss, student.trainable_variables)
            optimizer.apply_gradients(zip(grads, student.trainable_variables))

            losses.append(float(loss.numpy()))

        val_probs = model_probs(
            student,
            X_val,
            batch_size=512,
            temperature=1.0
        )

        val_pred = np.argmax(val_probs, axis=1)

        val_acc = float(np.mean(val_pred == y_val))
        val_f1 = float(f1_score(y_val, val_pred, average="macro"))

        if val_f1 > best_f1 + 1e-5:
            best_f1 = val_f1
            best_acc = val_acc
            best_weights = student.get_weights()

        lr_value = float(lr_schedule(optimizer.iterations).numpy())

        print(
            f"Student epoch {epoch + 1:03d}/{STUDENT_EPOCHS} | "
            f"loss={np.mean(losses):.4f} | "
            f"val_acc={val_acc:.4f} | "
            f"val_macroF1={val_f1:.4f} | "
            f"best_f1={best_f1:.4f} | "
            f"lr={lr_value:.2e}"
        )

    if best_weights is not None:
        student.set_weights(best_weights)

    print(f"Student restored best weights. Best val acc={best_acc:.4f}, best val F1={best_f1:.4f}")

    return student


def main():
    ensure_dir(FIGURES_DIR)

    print("=" * 72)
    print("MODEL 13.8")
    print("=" * 72)
    print("Feature mode:", FEATURE_MODE)
    print("Split protocol:", SPLIT_PROTOCOL)
    print("Output folder:", FIGURES_DIR)

    X_trials, y_trials, groups, meta = load_seediv_trials()

    print()
    print("Loaded EEG-only SEED-IV feature data")
    print("Total trials:", len(X_trials))
    print("Trial class counts:", np.bincount(y_trials, minlength=NUM_CLASSES))
    print("Example trial shape:", X_trials[0].shape, "= time_windows x channels x features")

    all_trial_idx = np.arange(len(X_trials))

    X_all, y_all_win, trial_ids_all, time_ids_all = trials_to_windows(
        X_trials,
        y_trials,
        all_trial_idx
    )

    (
        X_train,
        y_train,
        train_trial_ids,
        train_time_ids,

        X_val,
        y_val,
        val_trial_ids,
        val_time_ids,

        X_test,
        y_test,
        test_trial_ids,
        test_time_ids,
    ) = split_windows_directly(
        X_all,
        y_all_win,
        trial_ids_all,
        time_ids_all
    )

    X_train, X_val, X_test, scaler = normalize_windows(
        X_train,
        X_val,
        X_test
    )

    input_shape = X_train.shape[1:]

    print()
    print("Window input shape:", input_shape)
    print("Train windows:", X_train.shape, "counts:", np.bincount(y_train, minlength=NUM_CLASSES))
    print("Val windows:  ", X_val.shape, "counts:", np.bincount(y_val, minlength=NUM_CLASSES))
    print("Test windows: ", X_test.shape, "counts:", np.bincount(y_test, minlength=NUM_CLASSES))
    print()

    teacher1 = build_teacher(input_shape)

    print("Teacher 1 params:", teacher1.count_params())
    teacher1.summary()

    teacher1, teacher1_best_f1 = train_teacher_model(
        teacher1,
        X_train,
        y_train,
        X_val,
        y_val,
        teacher_name="Teacher 1"
    )

    teacher2 = build_teacher2(input_shape)

    print("Teacher 2 params:", teacher2.count_params())
    teacher2.summary()

    teacher2, teacher2_best_f1 = train_teacher_model(
        teacher2,
        X_train,
        y_train,
        X_val,
        y_val,
        teacher_name="Teacher 2"
    )

    student = build_student(input_shape)

    print("Student params:", student.count_params())
    student.summary()

    save_model_architecture(
        os.path.join(FIGURES_DIR, "model_13_8_architecture_summary.txt"),
        [teacher1, teacher2, student]
    )

    student = train_student_kd(
        student,
        teacher1,
        teacher2,
        X_train,
        y_train,
        X_val,
        y_val,
        teacher1_best_f1,
        teacher2_best_f1
    )

    print()
    print("=" * 72)
    print("FINAL TRAIN / VALIDATION / TEST EVALUATION")
    print("=" * 72)

    train_probs = model_probs(student, X_train, batch_size=512, temperature=1.0)
    val_probs = model_probs(student, X_val, batch_size=512, temperature=1.0)
    test_probs = model_probs(student, X_test, batch_size=512, temperature=1.0)

    metrics_rows = []

    metrics_rows.append(evaluate_and_save("train", y_train, train_probs))
    metrics_rows.append(evaluate_and_save("val", y_val, val_probs))
    metrics_rows.append(evaluate_and_save("test", y_test, test_probs))

    write_dict_csv(
        os.path.join(FIGURES_DIR, "metrics_summary_train_val_test.csv"),
        metrics_rows,
        fieldnames=[
            "split",
            "accuracy",
            "macro_f1",
            "weighted_f1",
            "mcc",
            "kappa",
            "specificity_macro",
            "auc_macro_ovr",
            "auc_weighted_ovr",
            "auc_micro",
            "auc_Neutral",
            "auc_Sad",
            "auc_Fear",
            "auc_Happy"
        ]
    )

    plot_final_accuracy_bar(
        metrics_rows,
        os.path.join(FIGURES_DIR, "train_val_test_accuracy.png")
    )

    final_summary = []
    final_summary.append("Model 13.8 Window Version Final Summary\n")
    final_summary.append(f"Feature mode: {FEATURE_MODE}\n")
    final_summary.append(f"Split protocol: {SPLIT_PROTOCOL}\n")
    final_summary.append(f"Input shape: {input_shape}\n")
    final_summary.append(f"Teacher 1 best validation macro-F1: {teacher1_best_f1:.4f}\n")
    final_summary.append(f"Teacher 2 best validation macro-F1: {teacher2_best_f1:.4f}\n\n")

    for row in metrics_rows:
        final_summary.append(
            f"{row['split'].upper()} | "
            f"Accuracy={row['accuracy']:.4f}, "
            f"Macro-F1={row['macro_f1']:.4f}, "
            f"MCC={row['mcc']:.4f}, "
            f"Kappa={row['kappa']:.4f}, "
            f"Specificity={row['specificity_macro']:.4f}, "
            f"AUC={row['auc_macro_ovr']:.4f}\n"
        )

    save_text(
        os.path.join(FIGURES_DIR, "final_summary.txt"),
        "".join(final_summary)
    )

    print()
    print("=" * 72)
    print("OVERALL FINAL ACCURACY")
    print("=" * 72)

    for row in metrics_rows:
        print(
            f"{row['split'].upper():5s} Accuracy: {row['accuracy']:.4f} | "
            f"Macro-F1: {row['macro_f1']:.4f} | "
            f"MCC: {row['mcc']:.4f} | "
            f"Kappa: {row['kappa']:.4f}"
        )

    print()
    print("Saved all requested outputs in:")
    print(FIGURES_DIR)


if __name__ == "__main__":
    main()