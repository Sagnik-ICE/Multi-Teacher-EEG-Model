# A Multi-Teacher Knowledge Distillation Model for Human Emotion Detection using EEG-Based Signals

## 1. Project Overview

This repository contains the Python code used for Model 13.8, a multi-teacher knowledge distillation framework for EEG-based emotion recognition. The model classifies four emotional states: **Neutral, Sad, Fear, and Happy**.

Two teacher networks learn from EEG features and guide the training of a smaller student network. The student learns from the correct emotion labels and the combined predictions of both teachers. After training, the student makes predictions without the teachers. This reduces the size of the model needed for inference.

The experiment uses the SEED-IV dataset. The code includes data loading, preprocessing, model training, knowledge distillation, evaluation, and result generation.

## 2. Code Information

The full experiment is implemented in one file:

**`Multi_Teacher_EEG_Model.py`**

It defines three networks:

| Network | Description | Total parameters |
| --- | --- | ---: |
| Teacher 1 | CNN with feature and channel attention, multi-scale separable convolutions, and three gated Transformer blocks | 3,619,378 |
| Teacher 2 | A smaller network with two gated Transformer blocks | 263,372 |
| Student | CNN with three residual separable convolution blocks, a residual LSTM, and a parallel feature path | 394,852 |

The main functions are listed below. All are in `Multi_Teacher_EEG_Model.py`.

| Function | Purpose |
| --- | --- |
| `load_seediv_trials()` | Reads the EEG feature files and their emotion labels. |
| `load_one_trial_multifeature()` | Combines the selected EEG feature groups for each trial. |
| `trials_to_windows()` | Converts trials into individual EEG windows. |
| `split_windows_directly()` | Creates the stratified train, validation, and test sets. |
| `normalize_windows()` | Fits a standard scaler on training data and applies it to all splits. |
| `build_teacher()` | Builds Teacher 1. |
| `build_teacher2()` | Builds Teacher 2. |
| `build_student()` | Builds the student network. |
| `train_teacher_model()` | Trains each teacher and restores its best validation weights. |
| `train_student_kd()` | Trains the student using class-weighted labels and the two teachers' soft predictions. |
| `augment_windows()` | Applies augmentation during student training. |
| `evaluate_and_save()` | Calculates performance metrics and saves evaluation results. |
| `main()` | Runs the full experiment in order. |

The two teachers are trained separately. Their weights are then frozen. Their soft predictions are combined using weights based on validation macro-F1. The student is trained using both the true labels and these combined predictions. The teachers are not used when the final student is evaluated.

## 3. Dataset Information

The experiments use **SEED-IV**, an EEG emotion recognition dataset developed at Shanghai Jiao Tong University. The implementation uses the prepared EEG feature files in the `eeg_feature_smooth` folder. It does not use the eye movement data or raw EEG recordings.

The dataset has 15 participants and three recording sessions. Each participant has 24 trials per session. The script expects 15 `.mat` files in each session folder. This gives **1,080 trials** across the three sessions.

The code combines four feature groups:

- `de_LDS`
- `de_movingAve`
- `psd_LDS`
- `psd_movingAve`

Each group provides five frequency-band features. The resulting input contains **62 EEG channels and 20 features per channel**. Each sample is one EEG window with shape **(62, 20)**. The PSD features are transformed with `log1p` after negative values are clipped to zero.

### Expected folder structure

```text
eeg_feature_smooth/
├── 1/
│   ├── subject_1.mat
│   └── ... (15 .mat files in total)
├── 2/
│   ├── subject_1.mat
│   └── ... (15 .mat files in total)
└── 3/
    ├── subject_1.mat
    └── ... (15 .mat files in total)
```

The file names above are examples. Keep the original SEED-IV file names. Each file must contain all four feature groups for trials 1 through 24, with fields such as `de_LDS1` and `psd_movingAve24`. The script assigns the session labels using its predefined SEED-IV label lists, so folders `1`, `2`, and `3` must contain the correct sessions.

**Important:** Use the complete 20-feature data for this experiment. The code includes a fallback to five DE features when fields are missing. That fallback is not part of the reported 20-feature run and may cause an input-shape mismatch if it occurs for only some trials. Check the files before training.

The dataset is not distributed with this repository. Its official description and access information are available from the [SEED-IV dataset page](https://bcmi.sjtu.edu.cn/home/seed/seed-iv.html). Follow the dataset provider's access and use conditions.

## 4. Requirements and Computing Environment

The code is written in **Python** and uses **TensorFlow/Keras** for training. The other main libraries are NumPy, SciPy, scikit-learn, and Matplotlib. The Python standard library is used for file handling and other tasks.

The following software and hardware details were recorded on **18 April 2026**.

### TensorFlow environment (`.tf210test`)

| Item | Recorded version or status |
| --- | --- |
| Python | 3.10.11 |
| `tensorflow-cpu` | 2.10.1 |
| `tensorflow-intel` | 2.10.1 |
| `tensorflow-directml-plugin` | 0.2.0.dev221020 |
| NumPy | 1.26.4 |
| GPU visible to TensorFlow | Yes, through DirectML |

The script also uses the `keras` package. Its separate installed version was not recorded. The exact installed versions of SciPy, scikit-learn, and Matplotlib were also not recorded. The table is an environment snapshot, not a complete package lockfile.

### Hardware

| Item | Configuration |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5050 Laptop GPU |
| CPU | Intel Core 5 210H |
| Memory | 16 GB DDR5 RAM |
| NVIDIA driver | 592.00 |

A separate main project environment (`.venv`) used Python 3.10.11 and PyTorch 2.11.0+cu130 with CUDA runtime 13.0. **PyTorch is not used by this script.** These PyTorch and CUDA versions should not be confused with the TensorFlow DirectML environment.

## 5. How to Run the Experiment

The instructions below are for Windows PowerShell. You need the SEED-IV `eeg_feature_smooth` data and the main Python file.

### Step 1. Open the project folder

Place `Multi_Teacher_EEG_Model.py` in a folder of your choice. Open that folder in PowerShell or the VS Code terminal.

### Step 2. Prepare the Python environment

If your TensorFlow environment is already set up, activate it. For an environment stored in the project folder, use:

```powershell
.\.tf210test\Scripts\Activate.ps1
```

If you are starting from a new installation, create a Python 3.10 environment first:

```powershell
py -3.10 -m venv .tf210test
.\.tf210test\Scripts\Activate.ps1
```

Install the required libraries in that environment. The recorded setup uses TensorFlow 2.10.1 and the DirectML plugin. You can use the following as a starting point:

```powershell
python -m pip install "tensorflow-cpu==2.10.1" "tensorflow-directml-plugin==0.2.0.dev221020" "numpy==1.26.4" scipy scikit-learn matplotlib
```

Package compatibility can depend on the local system. If installation reports a conflict, do not force incompatible versions. Check the TensorFlow and DirectML package requirements before proceeding. The exact set of working package versions should be verified in your own environment.

Check that the libraries can be imported:

```powershell
python -c "import tensorflow as tf, keras, numpy, scipy, sklearn, matplotlib; print('TensorFlow:', tf.__version__); print('NumPy:', numpy.__version__); print('GPU devices:', tf.config.list_physical_devices('GPU'))"
python -m pip check
```

A GPU is helpful but is not required by the Python code. GPU availability depends on the installed TensorFlow environment.

### Step 3. Set the data and output paths

Open `Multi_Teacher_EEG_Model.py`. Change these two values near the beginning of the file:

```python
DATASET_PATH = r"C:/path/to/eeg_feature_smooth"
RESULTS_DIR = r"C:/path/to/Model_13_8_Outputs"
```

Use paths that exist on your computer. The script creates the `Figures` subfolder for its results.

### Step 4. Run the code

```powershell
python Multi_Teacher_EEG_Model.py
```

The script runs the full experiment. It first loads and prepares the data. It then trains Teacher 1 and Teacher 2, followed by the student. Finally, it evaluates the student and saves the result files. Training time depends on the hardware.

## 6. Methodology and Training Settings

### Data preparation

The code converts the EEG trials into separate windows and assigns the emotion label of each trial to its windows. It uses a stratified random **window-level split** with random seed **42**.

| Data split | Number of windows | Share |
| --- | ---: | ---: |
| Training | 25,551 | 68% |
| Validation | 4,509 | 12% |
| Test | 7,515 | 20% |
| **Total** | **37,575** | **100%** |

The percentages describe the split settings. The window counts are from the recorded Model 13.8 run. A `StandardScaler` is fitted on the training windows only. The same scaler is used for validation and test windows. The script applies class-based sample weights during teacher and student training.

### Training configuration

| Setting | Value |
| --- | --- |
| Random seed | 42 |
| Emotion classes | 4 |
| Input shape | 62 × 20 |
| Teacher epochs | 50 per teacher |
| Student epochs | 40 |
| Teacher and student batch size | 256 |
| Optimizer | Adam |
| Gradient clip norm | 1.0 |
| Initial teacher learning rate | 0.0002 |
| Initial student learning rate | 0.00025 |
| Teacher learning rate schedule | ReduceLROnPlateau |
| Student learning rate schedule | Cosine decay |
| Label smoothing | 0.02 |
| L2 regularization coefficient | 0.00004 |
| Distillation temperature | 4.0 |
| Model selection | Highest validation macro-F1 |

Each teacher uses categorical cross-entropy with label smoothing. The learning rate is reduced when validation loss stops improving. The best teacher weights are selected using validation macro-F1.

### Knowledge distillation

The two teachers produce soft class probabilities using a temperature of **4.0**. Each teacher's contribution is proportional to its best validation macro-F1. This creates one combined soft target for the student.

The student loss combines a hard-label cross-entropy term and a temperature-scaled KL divergence term. L2 regularization is also included. When the best teacher validation macro-F1 is at least **0.70**, the hard-label loss receives a weight of **0.65**. Otherwise, it receives **0.88**. The soft-label term receives the remaining weight.

The student also uses augmentation during training. The operations include EEG channel masking, frequency-band masking, small Gaussian noise, and amplitude scaling. Each operation is considered separately with probability **0.30**. The teacher soft targets are computed from the original training windows before these augmentations.

The student weights with the best validation macro-F1 are restored before the final evaluation.

## 7. Outputs and Evaluation

The script saves its outputs in the `Figures` folder inside `RESULTS_DIR`.

| File or group | Contents |
| --- | --- |
| `metrics_summary_train_val_test.csv` | Accuracy, F1 scores, MCC, Cohen's kappa, specificity, and AUC values |
| `classification_report_train.txt`, `classification_report_val.txt`, `classification_report_test.txt` | Precision, recall, F1 score, and support for each class |
| `confusion_matrix_train.png`, `confusion_matrix_val.png`, `confusion_matrix_test.png` | Confusion matrices |
| `roc_curve_train.png`, `roc_curve_val.png`, `roc_curve_test.png` | ROC curves |
| `specificity_train.csv`, `specificity_val.csv`, `specificity_test.csv` | Class-level specificity and confusion counts |
| `model_13_8_architecture_summary.txt` | Model structures and parameter counts |
| `train_val_test_accuracy.png` | Accuracy across the three data splits |
| `final_summary.txt` | Summary of evaluation results |

The provided code does **not** save trained model checkpoint files. Reproducing the experiment requires running the training procedure again.

### Recorded Model 13.8 results

The recorded final student evaluation reported:

| Metric | Test result |
| --- | ---: |
| Accuracy | **98.46%** |
| Macro-F1 | **98.44%** |
| Matthews correlation coefficient | 0.9794 |
| Cohen's kappa | 0.9793 |
| Macro specificity | 99.48% |
| Macro one-vs-rest AUC | 99.95% |
| Test windows | 7,515 |

These values come from the recorded Model 13.8 run. They are not guaranteed to be identical on every rerun.

### Evaluation limitation

**This experiment uses a random window-level split.** Windows from the same participant or trial can appear in different splits. The reported accuracy is therefore a window-level result. It does not establish performance on unseen participants or unseen trials.

This distinction matters when comparing the result with studies that use subject-independent or trial-independent evaluation.

## 8. Citation

If you use this implementation in research, please cite the associated paper:

*A Multi-Teacher Knowledge Distillation Model for Human Emotion Detection using EEG-Based Signals.*

Please also cite the original SEED-IV dataset publication:

Zheng, W.-L., Liu, W., Lu, Y., Lu, B.-L., and Cichocki, A. (2019). **EmotionMeter: A Multimodal Framework for Recognizing Human Emotions.** *IEEE Transactions on Cybernetics, 49*(3), 1110–1122. https://doi.org/10.1109/TCYB.2018.2797176

Dataset information: https://bcmi.sjtu.edu.cn/home/seed/seed-iv.html
