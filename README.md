# Brain Tumor MRI Classification using CNN (PyTorch)

This project implements a **Convolutional Neural Network (CNN)** from scratch in **PyTorch**  
to classify MRI brain images as **Tumor** or **Non-Tumor** using a public Kaggle dataset.

---

## Dataset

Dataset: [Brain Tumor (Kaggle)](https://www.kaggle.com/datasets/jakeshbohaju/brain-tumor)

The dataset contains MRI images and a CSV file with two columns:
- `Image`: image filename
- `Class`: label (1 = Tumor, 0 = Non-Tumor)

---

## Features

- Custom PyTorch `Dataset` and `DataLoader`
- Data preprocessing and augmentation
- CNN architecture built from scratch
- Full training and validation loops
- Model checkpoint saving (`best_model.pth`)
- Visualization of loss and accuracy curves

---

## Model Architecture

```python
Conv2d(3,32,3,padding=1) → ReLU → MaxPool2d(2,2)
Conv2d(32,64,3,padding=1) → ReLU → MaxPool2d(2,2)
Conv2d(64,128,3,padding=1) → ReLU → MaxPool2d(2,2)
Flatten → Linear(128*16*16,128) → ReLU → Linear(128,2)
```