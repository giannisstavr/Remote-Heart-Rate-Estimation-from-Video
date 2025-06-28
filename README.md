# Remote Heart Rate Estimation from Video

## Master's Project

This repository contains the code for my Master's project, "Artificial Intelligence for remote heart rate estimation in video." The project focuses on developing and validating non-contact methods for continuous heart rate monitoring using facial video analysis, addressing limitations of traditional contact-based techniques.

The core contribution of this repository is the implementation of a **2D+1 Convolutional Neural Network (ConvLSTM) model** for remote photoplethysmography (rPPG). Such model was then compared to a custom-built pulsometer for validation and a mathematical estimation method.

## Table of Contents

- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Usage](#usage)
  - [Neural Network (2D+1 ConvLSTM)](#neural-network-2d1-convlstm)
- [Data](#data)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Project Overview

Heart rate (HR) is a crucial physiological signal, traditionally measured using contact-based methods like Electrocardiograms (ECGs) or Photoplethysmography (PPG).While accurate, these methods can cause discomfort or be impractical in certain scenarios (e.g., for newborns, burn victims). This project explores Remote Photoplethysmography (rPPG), which detects subtle skin color changes in video recordings caused by blood flow to estimate HR non-invasively.

## Key Features

* **Non-Contact Heart Rate Estimation:** Implementations of rPPG techniques for remote HR monitoring from video input.
* **Deep Learning Model:** A 2D+1 ConvLSTM neural network for robust spatiotemporal feature extraction and direct HR prediction from video frames.
* **Real-time Inference & Visualization:** Live demonstration capabilities with a user interface showing predicted heart rates and visual feedback.
* **Data Pre-processing and Correction:** Scripts for preparing diverse public datasets and implementing post-processing correction methods to improve accuracy.

### 2D+1 ConvLSTM Neural Network Architecture

The final neural network architecture for remote heart rate estimation is a 2D+1 ConvLSTM model. It is designed to capture subtle spatiotemporal features from facial video.

#### Spatial Feature Extraction (Encoder)
The model uses a `TimeDistributed` layer with a ResNet50 backbone (specifically, output from `conv4_block6_out`) as its spatial encoder. This ResNet50 model is pre-trained on large facial datasets to efficiently extract relevant facial features. The output from th 4th convolutional layer is used to preserve more high-level spatial features.

#### Temporal Processing (ConvLSTM Layers)
Two `ConvLSTM2D` layers, are used to process the sequence of spatial features. `BatchNormalization` is applied after each `ConvLSTM2D` layer to improve convergence and accuracy.

#### Channel Recalibration and Regression
A custom channel recalibration block (inspired by attention mechanisms) is applied after the ConvLSTM layers. This block helps the network suppress background noise and amplify pulse-related signals. Finally, a `GlobalAveragePooling2D` layer flattens the features, which are then passed through two `Dense` layers with ReLU activation (128 and 64 units) and a final `Dense` layer with linear activation to output the predicted heart rate.

The model is trained with a custom OneCycle Learning Rate Scheduler and `AdamW` optimizer for optimal performance and to mitigate overfitting.

The PURE dataset was used as a training set.

## Data

The project utilized and processed several public datasets, and also created a custom dataset. Due to ethical considerations and privacy safeguards, sensitive or identifiable data (including video recordings) are **not** uploaded to this public GitHub repository.

* **Utilized Public Datasets:** COHFACE , MAHNOB-HCI , PURE , UBFC-PHYS , UBFC-rPPG.
* **Custom Strathclyde Dataset:** Collected by project members to simulate real-world conditions for additional benchmarking.

## Results

* **Mean Absolute Error (MAE) Comparison (vs. Commercial Oximeter Ground Truth):**
    * Baseline Mathematical Model: 6.24 BPM (after correction)
    * Proposed Neural Network: 8.98 BPM (after correction)
* **Correction Methods Impact:** Post-processing correction strategies (linear and GMM-based) significantly improved the accuracy of both the baseline model and neural network models. The hybrid (Linear + GMM) correction often provided the best results for the baseline model, while linear correction was most effective for the neural network.
* **Environmental Challenges:** Models exhibited sensitivity to lighting variations, ROI obstruction, and speech-induced distortions, highlighting areas for future improvement.
