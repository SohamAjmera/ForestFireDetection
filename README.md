# Forest Fire Detection System 🔥🌲

This project is a **binary image classification system** that detects whether an image contains a **forest fire** or **no fire** using a **Convolutional Neural Network (CNN)** built with TensorFlow/Keras.

The core of the project is implemented in the Jupyter notebook:

> `ForestFireDetectionSystem.ipynb`

---

## 🚀 Project Overview

The goal of this project is to:
- Automatically **identify fire vs. no-fire** from forest images.
- Help in building an early-warning component for **wildfire monitoring systems**.
- Show a complete workflow from **data loading → preprocessing → model training → evaluation** using a real-world dataset.

The notebook:
1. **Downloads and prepares** a forest fire image dataset.
2. **Builds a DataFrame** of file paths and their labels (`FIRE`, `NO_FIRE`).
3. **Visualizes** class distribution and sample images.
4. **Creates image generators** with augmentation.
5. **Builds and trains a CNN model**.
6. **Saves the best model** for later inference.

---

## 📂 Dataset

The notebook uses a Kaggle dataset of forest fire images, accessed using `kagglehub`:

- Root path used in the notebook:
  ```python
  Fire_Dataset_Path = Path("../input/fire-dataset/fire_dataset")
