# Brain Tumor Classification Streamlit App

This project is a **Streamlit-based web application** for brain tumor classification using MRI images. It leverages multiple models including **CNN**, **SVM**, and **Random Forest** with **Genetic Algorithm optimization**.

---

## 🗂 Project Structure

```
C:.
│   .gitattributes
│   accuracy.png
│   CNN.py
│   data_augmentation.py
│   IRM_CNN.h5
│   loss.png
│   MRI_SVM_model.pkl
│   PostDist.png
│   PreDist.png
│   requirements.txt
│   RF_model.pkl
│   RF_with_GA.py
│   SVM.py
│   usi.py
|   README.md
│
+---brisc2025
│   \---classification_task
│       +---test
│       |   +---glioma
│       |   +---meningioma
│       |   +---no_tumor
│       |   \---pituitary
│       \---train
│           +---glioma
│           +---meningioma
│           +---no_tumor
│           \---pituitary
\---brisc2025_balanced_aug
    \---train
        +---glioma
        +---meningioma
        +---no_tumor
        \---pituitary
```

---

## ⚡ Features

* **MRI Brain Tumor Classification**: Glioma, Meningioma, No Tumor, Pituitary.
* **Models Implemented**:

  * Convolutional Neural Network (CNN)
  * Support Vector Machine (SVM)
  * Random Forest (RF)
  * Random Forest optimized with Genetic Algorithm (GA)
* **Data Augmentation**: Performed to balance classes and improve generalization.
* **Visualization**: Loss curves, accuracy plots, and distribution plots included.
* **Explainable AI**: SHAP and LIME visualizations implemented for CNN model predictions.

---

## 🛠 Installation

1. Clone the repository:

```bash
git clone https://github.com/NourBesrour/ProjetDeSession_XAI
cd ProjetDeSession_XAI

```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Streamlit App

Run the main Streamlit app:

```bash
streamlit run usi.py
```

This will launch the app in your browser where you can:

* Upload MRI images.
* Predict tumor type.
* Visualize model explanations.

---

## 🧠 Model Details

| Model   | File                | Description                                                                                     |
| ------- | ------------------- | ----------------------------------------------------------------------------------------------- |
| CNN     | `IRM_CNN.h5`        | Trained on augmented MRI images using Keras.                                                    |
| SVM     | `MRI_SVM_model.pkl` | SVM classifier trained on HOG features extracted from MRI images.                               |
| RF      | `RF_model.pkl`      | Random Forest classifier on extracted features.                                                 |
| RF + GA | `RF_with_GA.py`     | Random Forest optimized with Genetic Algorithm for feature selection and hyperparameter tuning. |

---

## 📊 Visualizations

* `accuracy.png` — Model accuracy comparison.
* `loss.png` — CNN training loss curves.
* `PreDist.png` / `PostDist.png` — Distribution of classes before and after augmentation.
* `notes.txt` — Additional notes about model training.

---

## 🧹 Data Structure

* `brisc2025/classification_task/train` — Training images split by tumor type.
* `brisc2025/classification_task/test` — Testing images split by tumor type.
* `brisc2025_balanced_aug/train` — Augmented and balanced training dataset.

---

## 📌 Notes

* Ensure that all dependent models (`.h5`, `.pkl`) are in the project root before running the app.
* Data augmentation scripts are in `data_augmentation.py`.
* CNN training script is `CNN.py`.
* SVM training script is `SVM.py`.
* Random Forest GA script is `RF_with_GA.py`.

---


