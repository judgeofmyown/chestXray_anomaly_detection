# 🩺 Chest X-ray Anomaly Detection using Deep Learning  
[![Python](https://img.shields.io/badge/Python-3.9-blue?logo=python)](https://www.python.org/) 
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13-red?logo=pytorch)](https://pytorch.org/)


Detecting anomalies in chest X-ray images using deep learning with a focus on **unsupervised/self-supervised techniques** — a practical response to the scarcity of labeled medical data.

---

## 🧠 Motivation

In medical imaging, especially radiology, large annotated datasets are rare and expensive to obtain. This project aims to:

- Explore **unsupervised anomaly detection** in chest X-rays.
- Provide a modular, reproducible pipeline for **real-world medical ML problems**.
- Lay the foundation for using **pretrained models** and **representation learning** in medical diagnostics.

---

## 🏗️ Architecture Overview

The project pipeline includes:

1. 📥 **Data loading & transformation**
2. 🔍 **Feature extraction using pretrained CNNs**
3. 🎯 **Anomaly detection via clustering or reconstruction error**
4. 📈 **Evaluation and logging**

---

## 📁 Project Structure

```text
├── chestXrayData.py       # Data loading and transformation pipeline
├── modelPreTrained.py     # Pretrained model setup (e.g., ResNet/ViT)
├── train.py               # Training logic and loss computation
├── training_script.py     # Entry point for training
├── logtest.py             # Logs & testing utilities
├── requirements.txt       # Dependencies
└── README.md              # This file
