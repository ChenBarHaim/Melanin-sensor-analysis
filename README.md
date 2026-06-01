# Melanin-sensor-analysis

# Wearable Melanin-ChCl Humidity Sensor: Respiratory Signal Processing & Apnea Detection

This repository contains the signal processing pipeline, data analysis, and automated detection algorithms developed for a novel, wearable, inkjet-printed respiratory sensor. The project benchmarks a sustainable bio-derived sensor against clinical gold standards and automates the detection of obstructive sleep apnea.

## 📌 Project Overview & Motivation
Sleep apnea affects nearly 1 billion people worldwide, yet most remain undiagnosed due to the discomfort of traditional, bulky polysomnography (PSG) equipment. 

This project introduces a data pipeline for a flexible, skin-conformable humidity sensor based on a bio-derived synthetic melanin-choline chloride (ChCl) matrix. The sensor tracks respiration by measuring resistance changes modulated by breath hydration, offering a high-speed, non-invasive alternative for clinical and home sleep monitoring.

## 🛠️ Key Engineering & Data Features
*   **Signal Conditioning & Artifact Removal:** Implementation of filtering algorithms (including rolling smoothing windows) to handle motion artifacts and baseline drift in raw resistance data (MΩ).
*   **Respiratory Rate (BPM) Quantification:** Automated peak detection algorithms to extract breath-by-breath intervals and calculate dynamic Respiratory Rate (BPM).
*   **Automated Apnea Detection:** A dynamic-thresholding state machine designed to accurately isolate periods of cessation of breath (Apnea) from normal breathing.
*   **Statistical Benchmarking & Validation:** Comparative analysis using Bland-Altman plots and Confusion Matrices to validate performance against medical-grade reference sensors.

## 📊 Main Results & Analysis

### 1. Verification Against Clinical Gold Standards
The Melanin-ChCl sensor's performance was rigorously benchmarked against a commercial **Thermo-Can thermistor** (the clinical gold standard). 
*   **BPM Agreement:** A rolling 10-second window Bland-Altman analysis demonstrated moderate and consistent agreement between the methods, exhibiting a minor mean bias of -2.04 BPM.
*   **Signal Equivalence:** The processed ionic/electronic conductivity signals show exceptional alignment with traditional thermal and airflow sensors, clearly capturing fine respiratory dynamics.

### 2. Automated Sleep Apnea Detection Performance
By utilizing an automated dynamic-threshold algorithm across multiple evaluation segments, the pipeline achieved high-fidelity diagnostics:
*   **Sensitivity:** **100%** (Zero false negatives; correctly identified every apnea event).
*   **Accuracy:** **95.5%** overall system accuracy.

| Metrics Evaluated | Validation Results |
| ------------- | ------------- |
| **True Positive (Apnea Detected Correctly)** | 13.6% |
| **True Negative (Normal Breathing Detected)** | 81.8% |
| **False Negative (Missed Apnea)** | 0.0% |
| **False Positive (False Alarm)** | 4.5% |

### 3. Multi-Channel Electrophysiology Synchronization
The repository also includes scripts validating the synchronization of this respiratory airflow data with multi-channel electrophysiological signals, including:
*   Electroencephalography (**EEG**)
*   Facial Electromyography (**EMG**)
*   Electrooculography (**EOG**)

This multi-modal synchronization lays the groundwork for a comprehensive, single-patch sleep assessment framework.

## 💻 Tech Stack
*   **Language:** Python / MATLAB
*   **Libraries used:** NumPy, SciPy (Signal Processing toolbox), Matplotlib, Pandas, Scikit-learn (for confusion matrix and statistical evaluation).

## 📂 Repository Structure
*   `/src` or `/scripts`: Source code for filtering, peak detection, and the apnea state-machine.
*   `/plots` or `/images`: Visual outputs, including the confusion matrix and Bland-Altman plots.
*   `notebook.ipynb` (Optional): A walkthrough Jupyter Notebook demonstrating the pipeline from raw sensor log to final apnea classification.

---
*Note: This research was conducted in collaboration between Tel Aviv University and the Karlsruhe Institute of Technology (KIT), Germany.*
