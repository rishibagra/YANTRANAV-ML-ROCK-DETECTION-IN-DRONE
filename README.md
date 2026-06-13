# Mineral Detection using Computer Vision & Deep Learning

## Overview

This project presents a hybrid **Computer Vision** and **Deep Learning** pipeline for detecting Iron Oxide-rich regions from geological imagery. The system combines deep feature extraction, similarity-based matching, template matching, and post-processing techniques to identify mineral targets with improved accuracy and localization.

The solution is designed to assist in:

- Mineral exploration
- Geological surveys
- Mining applications
- Automated rock surface analysis

---

## Features

- Deep Learning-based feature extraction
- Cosine similarity matching in latent feature space
- Classical Computer Vision template matching
- Non-Maximum Suppression (NMS) for overlap removal
- Automated mineral target localization
- Hybrid Deep Learning + Traditional CV architecture
- Accurate detection of amorphous Iron Oxide textures

---

## Methodology

### 1. Deep Feature Extraction

A pretrained deep neural network is used to extract high-dimensional feature embeddings from geological images.

### 2. Feature Similarity Analysis

Cosine similarity is used to compare extracted features with reference Iron Oxide samples.

**Threshold Used:**

```text
Cosine Similarity ≥ 0.70
```

This enables robust matching of irregular and amorphous mineral textures.

### 3. Structural Pattern Matching

Template matching is performed using OpenCV's normalized cross-correlation method:

```python
cv2.TM_CCOEFF_NORMED
```

**Threshold Used:**

```text
Match Confidence ≥ 0.45
```

This helps identify geometric and structural similarities across varying scales.

### 4. Post-Processing

Non-Maximum Suppression (NMS) is applied to eliminate duplicate detections.

**IoU Threshold:**

```text
IoU = 0.20
```

This ensures clean and precise boundary localization.

---

## Detection Pipeline

```text
Input Geological Image
          │
          ▼
Deep Feature Extraction
          │
          ▼
Cosine Similarity Matching
          │
          ▼
Template Matching
          │
          ▼
Candidate Detections
          │
          ▼
Non-Maximum Suppression
          │
          ▼
Final Mineral Localization
```

---

## Technologies Used

### Programming Language

- Python

### Libraries & Frameworks

- OpenCV
- NumPy
- Scikit-Learn
- TensorFlow / PyTorch
- Matplotlib

---

## Key Parameters

| Component | Value |
|------------|--------|
| Cosine Similarity Threshold | 0.70 |
| Template Matching Confidence | 0.45 |
| NMS IoU Threshold | 0.20 |

---

## Applications

- Mineral Exploration
- Geological Survey Automation
- Mining Industry
- Remote Sensing Analysis
- Resource Mapping
- Terrain Assessment

---

## Future Improvements

- Real-time detection using YOLO
- Multi-mineral classification
- Drone-based geological surveying
- GIS integration
- 3D mineral mapping
- Edge deployment for field applications

---

## Results

- Successfully combined Deep Learning and Classical Computer Vision techniques.
- Achieved robust mineral target identification in geological imagery.
- Reduced duplicate detections using NMS-based refinement.
- Improved localization accuracy through hybrid feature matching.

---

## Project Structure

```text
├── data/
├── models/
├── notebooks/
├── src/
│   ├── feature_extraction.py
│   ├── template_matching.py
│   ├── nms.py
│   └── detection_pipeline.py
├── results/
├── requirements.txt
└── README.md
```

---

## Author

### Rishi Bagra

- B.Tech, Electronics & Communication Engineering
- National Institute of Technology (NIT)
- Interests: Machine Learning, Computer Vision, AI Systems, Industrial Automation

---

## License

This project is licensed under the MIT License.
