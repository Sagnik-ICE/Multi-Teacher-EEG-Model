# A Multi-Teacher Knowledge Distillation Model for Human Emotion Detection using EEG-Based Signals

## Overview

This project provides the code for a four-class emotion recognition model using electroencephalogram (EEG) features. It uses two teacher networks to train a smaller student network through knowledge distillation. After training, only the student is needed to predict emotions.

The four classes are **Neutral**, **Sad**, **Fear**, and **Happy**. The implementation corresponds to the final experiment, **Model 13.8**.

## Code

The main file is `Multi_Teacher_EEG_Model.py`. It includes data loading, preprocessing, model training, evaluation, and figure generation.

The code builds three networks:

- **Teacher 1:** A CNN with feature and channel attention, multi-scale convolution, and Transformer blocks.
- **Teacher 2:** A smaller Transformer-based network.
- **Student:** A CNN with separable convolution blocks and a residual LSTM path.

The code also performs weighted teacher fusion and knowledge distillation. Both teachers are frozen before the student is trained. At test time, predictions come from the trained student alone.

## Dataset

The experiments use the **SEED-IV** EEG dataset. The code reads the prepared EEG feature files from the `eeg_feature_smooth` folder. It does not use the eye-tracking data.

The dataset contains 15 participants, three sessions per participant, and 24 trials per session. The code loads **1,080 trials** in total. Each session folder is expected to contain 15 MATLAB `.mat` files.

The input features come from four sets: `de_LDS`, `de_movingAve`, `psd_LDS`, and `psd_movingAve`. Each set contains five frequency-band features. These are combined into **20 features per EEG channel**. Each input window has shape **62 × 20**.

Place the dataset in a folder with this structure:

```text
eeg_feature_smooth/
├── 1/
│   └── [15 subject .mat files]
├── 2/
│   └── [15 subject .mat files]
└── 3/
    └── [15 subject .mat files]
```

Each `.mat` file must contain the indexed feature fields for trials 1 to 24. Use the complete four-feature version of the files. If a required feature field is missing, the script may fall back to a five-feature input for that trial.

The dataset is not included with this code. Use it under the conditions set by the SEED-IV dataset provider.

## Software and hardware

The project uses **Python 3.10.11**. The model is implemented with **TensorFlow and Keras**. Data processing and evaluation also use NumPy, SciPy, scikit-learn, and Matplotlib.

The following environment details were recorded on **18 April 2026**.

### TensorFlow DirectML environment (`.tf210test`)

This is the recorded TensorFlow environment. It is the relevant environment for the model code.

| Software | Recorded version or status |
|---|---|
| Python | 3.10.11 |
| `tensorflow-cpu` | 2.10.1 |
| `tensorflow-intel` | 2.10.1 |
| `tensorflow-directml-plugin` | 0.2.0.dev221020 |
| NumPy | 1.26.4 |
| TensorFlow GPU visibility | Yes, through DirectML |

The code also imports Keras directly. Its separate package version was not included in the recorded environment details. SciPy, scikit-learn, and Matplotlib are required, but their exact installed versions were not supplied.

### Main project environment (`.venv`)

| Software | Recorded version or status |
|---|---|
| Python | 3.10.11 |
| PyTorch | 2.11.0+cu130 |
| PyTorch CUDA runtime | 13.0 |
| PyTorch CUDA available | Yes |
| NVIDIA driver | 592.00 |

**PyTorch is not used in `Multi_Teacher_EEG_Model.py`.** The PyTorch and CUDA information describes the separate main project environment. It should not be treated as the TensorFlow runtime configuration. The TensorFlow environment detects GPU devices through DirectML.

The computer has an **NVIDIA GeForce RTX 5050 Laptop GPU**, an **Intel Core 5 210H processor**, and **16 GB DDR5 RAM**.

## How to run

**1. Open the project folder and use a TensorFlow environment.** The code needs TensorFlow/Keras, NumPy, SciPy, scikit-learn, and Matplotlib. On Windows, activate the existing `.tf210test` environment if it contains these libraries. For example, if the environment is in the project folder, use PowerShell:

```powershell
.\.tf210test\Scripts\Activate.ps1
```

Check the Python and TensorFlow versions before running the experiment:

```powershell
python --version
python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"
```

These commands inspect the active environment. They do not install or change packages. If the environment is stored elsewhere, activate it using its actual path. The recorded version list above is an environment snapshot rather than a complete installation lockfile.

**2. Set the dataset and output paths.** Open `Multi_Teacher_EEG_Model.py` and update these lines to match your computer:

```python
DATASET_PATH = r"C:/path/to/eeg_feature_smooth"
RESULTS_DIR = r"C:/path/to/Model_13_8_Outputs"
```

The script creates the `Figures` folder inside `RESULTS_DIR`.

**3. Run the experiment.** Keep the default model settings if you want to follow the reported experiment.

```powershell
python Multi_Teacher_EEG_Model.py
```

The script loads and splits the data, trains Teacher 1 and Teacher 2, trains the student through knowledge distillation, and evaluates the student on the training, validation, and test sets. GPU use depends on the active TensorFlow environment and device support.

## Experimental settings

| Setting | Value |
|---|---|
| Random seed | 42 |
| Data split | Stratified random window-level split |
| Training / validation / test | 68% / 12% / 20% |
| Teacher training | 50 epochs each |
| Student training | 40 epochs |
| Batch size | 256 |
| Optimizer | Adam with gradient clip norm 1.0 |
| Initial teacher learning rate | 0.0002 |
| Initial student learning rate | 0.00025 |
| Teacher learning-rate schedule | ReduceLROnPlateau |
| Student learning-rate schedule | Cosine decay |
| Label smoothing | 0.02 |
| L2 regularization coefficient | 0.00004 |
| Distillation temperature | 4.0 |
| Student augmentation probability | 0.30 per augmentation operation |
| Model selection | Best validation macro-F1 |

The feature values are standardized using a scaler fitted **only on the training set**. The student uses both ground-truth labels and the weighted soft predictions of the two teachers. Teacher weights are based on their validation macro-F1 scores. The hard-label loss weight is **0.65** when the best teacher validation macro-F1 is at least 0.70. Otherwise, the code uses **0.88**.

During student training, the code applies channel masking, frequency-band masking, small random noise, and scaling. Teacher soft targets are calculated from the original training windows before these changes.

## Output files

The script writes its results to the `Figures` folder under `RESULTS_DIR`. Important outputs include:

- `metrics_summary_train_val_test.csv` — accuracy, F1, MCC, kappa, specificity, and AUC values.
- `classification_report_train.txt`, `classification_report_val.txt`, `classification_report_test.txt` — class-level results.
- `confusion_matrix_train.png`, `confusion_matrix_val.png`, `confusion_matrix_test.png` — confusion matrices.
- `roc_curve_train.png`, `roc_curve_val.png`, `roc_curve_test.png` — ROC plots.
- `specificity_train.csv`, `specificity_val.csv`, `specificity_test.csv` — class-level specificity results.
- `model_13_8_architecture_summary.txt` — model summaries and parameter counts.
- `train_val_test_accuracy.png` and `final_summary.txt` — overall results.

The provided script restores the best validation weights in memory. **It does not save trained model checkpoints**. Run the script to train and evaluate the networks again.

## Reported result and evaluation scope

In the recorded Model 13.8 run, the distilled student achieved **98.46% test accuracy** and **98.44% test macro-F1** on **7,515 test windows**. The student has **394,852 parameters**. Teacher 1 and Teacher 2 have **3,619,378** and **263,372** parameters, respectively.

**Important:** Windows are split at random after they are extracted from the trials. Windows from the same trial or participant can appear in more than one split. These results are therefore a **window-level benchmark**. They must not be described as trial-independent or subject-independent results. Small differences between reruns may occur because of software, hardware, and training randomness.

## Citation

If you use this code in research, please cite the associated article:

*A Multi-Teacher Knowledge Distillation Model for Human Emotion Detection using EEG-Based Signals.*

Please also cite the SEED-IV dataset publication recommended by its creators:

Wei-Long Zheng, Wei Liu, Yifei Lu, Bao-Liang Lu, and Andrzej Cichocki. **“EmotionMeter: A Multimodal Framework for Recognizing Human Emotions.”** *IEEE Transactions on Cybernetics*, 49(3), 1110–1122, 2019. https://doi.org/10.1109/TCYB.2018.2797176
