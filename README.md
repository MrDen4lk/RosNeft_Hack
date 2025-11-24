# 🗺️ RosNeft Hack: Semantic Segmentation of Schematics

![Python 3](https://img.shields.io/badge/python-3-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c?style=flat&logo=pytorch)
![ONNX](https://img.shields.io/badge/ONNX-Runtime-blue?style=flat&logo=onnx)
![Metric](https://img.shields.io/badge/IoU-0.96-brightgreen)

A computer vision solution for semantic segmentation of technical schematic symbols (40 classes), developed during the RosNeft Hackathon.

## 📋 Task Overview

The goal was to develop an algorithm to automatically recognize and segment specific symbols and designations on technical drawings and engineering schematics.

* **Input:** Technical schematic images.
* **Classes:** 40 classes + 1 background class.
* **Resolution:** 600x600 px.
* **Dataset:** 10,000 samples.

## 🛠️ Approach & Architecture

### Model Selection: U-Net++ with EfficientNet-B0 & Self-Attention
After conducting several experiments, **U-Net++** was selected as the final architecture.

* **Encoder (Backbone):** `EfficientNet-B0` (Pre-trained on ImageNet). Chosen for its optimal balance between training speed and feature extraction capability.
* **Decoder:** U-Net++ with **Nested Skip Pathways**, allowing the model to capture fine-grained details of small objects (lines, symbols) typical for technical drawings.
* **Attention Mechanism:** A **Self-Attention** block was integrated into the bottleneck. This allows the model to capture global context and long-range dependencies within the schematics.

### Why not Transformers (SegFormer)?
We attempted to train **SegFormer**. However, due to the specific nature of the data (sparse schematic lines vs. natural textures) and the limited dataset size (10k), the Transformer-based approach showed slower convergence and lower accuracy compared to the CNN-based approach with strong inductive bias.

### Training Details
* **Loss Function:** `DiceCELoss` (Dice Loss + Cross Entropy). This combination effectively handled the class imbalance problem (small symbols vs. large background area) and ensured sharp segmentation boundaries.
* **Target Metric:** `IoU` (Intersection over Union), calculated excluding the background class (Index 0).
* **Hardware:** Training was performed on **Nvidia RTX 5060 Ti**.

## 📊 Results

The model achieved high accuracy on the validation set.

| Metric | Score |
| :--- | :--- |
| **IoU (no background)** | **0.96** |
| Input Resolution | 600x600 |
| Number of Classes | 40 |

## 💻 Inference & Deployment

For production-ready inference, the model was converted to **ONNX** format. This reduced dependencies (removing the need for full PyTorch) and enabled hardware acceleration.
