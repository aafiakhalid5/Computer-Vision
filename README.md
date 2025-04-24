# 📸 Computer Vision Project – Winter Term 2024/25

This repository contains projects developed for the Computer Vision course at FAU Erlangen-Nürnberg. The course spans multiple exercises across different computer vision topics, each in its own module for clarity and modularity.

---

## 📁 Project Modules

Each directory corresponds to a different core area in computer vision, based on the lecture and exercise structure of the course.

### 🔬 Image Processing on Grayscale and Distance Images
Implements distance-based box detection using ToF sensor data. Core topics:
- Image and point cloud visualization
- RANSAC-based plane fitting
- Morphological operations for mask refinement

📁 `Image Processing on Grayscale and Distance Images/`

---

### 🎨 Demosaicing & HDR
Pipeline for processing raw sensor data:
- Bayer pattern identification
- Simple demosaicing
- Gamma correction and white balancing
- Exposure-based linearity test
- HDR composition from multiple exposures

📁 `Demosaicing & HDR/`

---

### ✍️ Writers Retrieval
Writer identification using local image descriptors and VLAD encoding:
- Codebook generation with MiniBatchKMeans
- VLAD feature aggregation
- SVM-based exemplar classification

📁 `Writers Retrieval/`

---

### 🧑‍🤝‍🧑 Face Recognition
Video-based face recognition pipeline:
- Face detection and tracking (MTCNN, template matching)
- Face alignment and embedding extraction (FaceNet)
- Closed-set and open-set face identification

📁 `Face Recognition/`

---

### 🏛️ Computer Vision in Humanities
Selective search for object proposals in historical domains:
- Region segmentation using Felzenszwalb
- Similarity-based region merging
- Final region proposals used in cultural artifact detection

📁 `Computer Vision in Humanities/`

---

## 🚀 Getting Started

Python ≥ 3.9 is recommended. Install required packages per folder's `requirements.txt` if available.

```bash
pip install -r requirements.txt
```

## 📜 License

This project is licensed under the Apache 2.0 License.

---

## 🤝 Acknowledgements

Project supervised by:
- Thomas Gorges
- Mathias Seuret
- Vincent Christlein
- Thomas Köhler
- Mathias Zinnen

Materials based on course content from the Pattern Recognition Lab at FAU.
