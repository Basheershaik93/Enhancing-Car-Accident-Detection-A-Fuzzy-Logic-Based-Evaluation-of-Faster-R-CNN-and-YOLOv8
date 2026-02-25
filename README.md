# Enhancing Car Accident Detection: A Fuzzy Logic-Based Evaluation of Faster R-CNN and YOLOv8

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/Basheershaik93/Enhancing-Car-Accident-Detection-A-Fuzzy-Logic-Based-Evaluation-of-Faster-R-CNN-and-YOLOv8/blob/main/LICENSE)
[![Conference: ICCCES 2026](https://img.shields.io/badge/Conference-ICCCES%202026-blue)](https://icoces.com/)
[![IEEE Xplore](https://img.shields.io/badge/IEEE-Xplore-Pending-orange)](https://ieeexplore.ieee.org/)

This repository contains the implementation, datasets, results, and supporting materials for the research paper:

**"Enhancing Car Accident Detection: A Fuzzy Logic-based Evaluation of Faster R-CNN and YOLOv8"**

presented at the **5th International Conference on Communication, Computing and Electronics Systems (ICCCES-2026)**.

The work proposes a hybrid accident detection framework that fuses predictions from **YOLOv8** and **Faster R-CNN** using **fuzzy logic** to improve robustness, reduce false positives, and deliver more reliable real-time decisions in challenging conditions (low light, occlusion, overlapping objects).

## Abstract

Real-time accident detection is essential for enhancing road safety and enabling quick emergency response. Models such as Faster R-CNN and YOLOv8 already offer strong performance, achieving detection accuracies of 80.1% and 99.5% (mAP@0.5), respectively. However, in challenging real-world scenarios—such as low visibility, occlusions, and overlapping objects—these models may produce inconsistent or conflicting predictions.

To overcome this limitation, we propose a hybrid accident detection system that combines YOLOv8 and Faster R-CNN using a fuzzy logic-based decision mechanism. Confidence scores from both models are processed through fuzzy membership functions—**Ignore**, **Uncertain**, and **Strong Detection**—and a rule-based inference engine to produce a more stable and interpretable final output.

Experiments on a custom dataset containing **Car**, **Car Accident**, and **Fire Accident** classes show that the hybrid approach achieves an improved mAP@0.5 of **90.3%**, reduces false positives, and delivers more reliable classification. These results demonstrate that fuzzy logic fusion effectively enhances the consistency and robustness of accident detection, making the system suitable for intelligent surveillance and real-time traffic monitoring.

**Index Terms**: YOLOv8, Faster R-CNN, Accident Detection, Fuzzy Logic, Intelligent Transportation Systems, Real-Time Surveillance.

## Paper

- **Title**: Enhancing Car Accident Detection: A Fuzzy Logic-based Evaluation of Faster R-CNN and YOLOv8  
- **Conference**: 5th International Conference on Communication, Computing and Electronics Systems (ICCCES-2026)  
- **Authors**: Shaik Basheer, Chanchal Biswas, Dunga Priyatham, Shaik Abdul Karim, Abdul Rehan  
- **Preprint**: [230.pdf](./230.pdf) (included in this repository)  
- **IEEE Xplore**: Publication pending (conference held recently; DOI forthcoming)

**Citation** (update DOI once available):

@inproceedings{
  author    = {Shaik Basheer and Chanchal Biswas and Dunga Priyatham and Shaik Abdul Karim and Abdul Rehan},
  title     = {Enhancing Car Accident Detection: A Fuzzy Logic-based Evaluation of Faster R-CNN and YOLOv8},
  booktitle = {2026 5th International Conference on Communication, Computing and Electronics Systems (ICCCES)},
  year      = {2026},
  pages     = {1527--1532},
  publisher = {IEEE},
  doi       = {TBD}
}

## Repository Structure

├── Dataset/
│   └── high_quality_pictures.zip          # Custom dataset (~366 MB – unzip before use)

├── ICCES-1032 Original Source File/       # Original conference submission materials

├── Results/                               # Model outputs, confusion matrices, visualizations, metrics

├── 230.pdf                                # Conference paper PDF (ICCCES 2026)

├── FUZZY_Logic.ipynb                      # Fuzzy logic fusion & decision engine

├── Faster_R_CNN.ipynb                     # Faster R-CNN training, evaluation & inference

├── Yolo_V8_training_(1).ipynb             # YOLOv8 training pipeline & experiments

├── confidence_of_the_both_models_.ipynb   # Confidence score analysis & model comparison

├── .gitignore                             # Ignore rules for caches, temp files, large binaries

├── .gitattributes                         # Git LFS tracking rules for large files (*.zip)

└── README.md                              # This file


## Installation

**Prerequisites**

Python 3.8+
Git LFS (for handling large dataset files): Install via git-lfs.com

**Setup**

**1. Clone the repository:**

git clone https://github.com/Basheershaik93/Enhancing-Car-Accident-Detection-A-Fuzzy-Logic-Based-Evaluation-of-Faster-R-CNN-and-YOLOv8.git
cd Enhancing-Car-Accident-Detection-A-Fuzzy-Logic-Based-Evaluation-of-Faster-R-CNN-and-YOLOv8

**2. Install Git LFS (for the large dataset zip)**

git lfs install
git lfs pull

**3. Install dependencies**

pip install -r requirements.txt  # If requirements.txt is added; otherwise, install manually
 Note: Required packages: torch, ultralytics (for YOLOv8), torchvision (for Faster R-CNN), scikit-fuzzy, numpy, opencv-python, matplotlib, pandas.
 
**Usage**

**1. Unzip Dataset:**

  unzip Dataset/high_quality_pictures.zip -d Dataset/
  
**2. Train Models**

 . Run Yolo_V8_training_(1).ipynb for YOLOv8.
 . Run Faster_R_CNN.ipynb for Faster R-CNN.
 
**3. Fuzzy Logic Confusion:**

. Use FUZZY_Logic.ipynb to process confidence scores and generate fused outputs.

**4. Evaluation:**

. Run confidence_of_the_both_models_.ipynb to compare mAP@0.5 and other metrics.
   Example command for inference (adapt from notebooks):
    python infer.py --model yolov8.pt --source video.mp4
    

## Results

| Model              | mAP@0.5 | False Positives | Inference Speed (ms) | Notes                              |
|--------------------|---------|-----------------|----------------------|------------------------------------|
| Faster R-CNN       | 80.1%   | High            | ~150                 | Strong on precision, slower        |
| YOLOv8             | 99.5%   | Moderate        | ~20                  | Very fast, higher false positives  |
| **Hybrid (Fuzzy)** | **90.3%** | **Low**       | ~35                  | Best balance – reduced false alarms by ~40% in low-visibility scenarios |

The hybrid approach reduces false positives by 40% in low-visibility scenarios while maintaining real-time performance.

**Contributors**

Shaik Basheer

Chanchal Biswas 

Dunga Priyatham 

Shaik Abdul Karim

Abdul Rehan 

All affiliated with Dept. of Computer Science and Engineering, SRM University AP, Neerukonda, Andhra Pradesh, India.

**License**
This project is licensed under the MIT License - see the LICENSE file for details.

**Acknowledgments**

Presented at ICCCES-2026 .

Thanks to SRM University AP for support.

Models built using Ultralytics YOLOv8 and PyTorch Faster R-CNN implementations.
