#  Week 2 Work: Image Processing & Medical Imaging Dataset Exploration

##  Objective
The objective of this week’s work is to:
1. Understand basic **image processing techniques** using Python.
2. Explore and critically analyze **real-world medical imaging datasets** used in machine learning and healthcare applications.

---

##  Part 1: Image Processing in Python

###  Tools & Libraries Used
- **OpenCV (cv2)** – Image loading, resizing, edge detection, filtering
- **NumPy** – Numerical operations on image matrices
- **Matplotlib** – Image and histogram visualization
- **scikit-image** – Noise generation utilities

---

###  Image Processing Steps Performed

#### 1. Image Loading
Two images were loaded using OpenCV and converted from **BGR to RGB** format for correct visualization.

#### 2. Grayscale Conversion
- RGB images were converted to grayscale.
- Grayscale images reduce computational complexity and are commonly used in medical imaging.

#### 3. Histogram Visualization
- Pixel intensity histograms were plotted.
- Histograms help analyze contrast, brightness, and intensity distribution.

#### 4. Image Resizing
- Images were resized to a fixed resolution (256 × 256).
- Resizing ensures consistent input dimensions for machine learning models.

#### 5. Edge Detection
- Canny Edge Detection algorithm was applied.
- Edges highlight anatomical structures such as bones and lesions.

#### 6. Noise Addition
- Gaussian noise was artificially added to simulate real-world sensor noise.

#### 7. Noise Removal
- Gaussian Blur filter was applied to reduce noise while preserving structure.


::contentReference[oaicite:0]{index=0}


---

##  Part 2: Medical Imaging Dataset Exploration

### Dataset 1: Chest X-ray Dataset (NIH)

####  Source
Provided by **:contentReference[oaicite:1]{index=1}**

####  Basic Information
- **Type:** Chest X-ray (Radiography)
- **Images:** ~112,000
- **Patients:** ~30,000
- **Labels:** 14 disease categories  
  (Pneumonia, Cardiomegaly, Edema, Fibrosis, etc.)

####  Dataset Imbalance
- Highly imbalanced dataset
- “No Finding” class dominates
- Rare diseases have very few samples

####  Challenges
- Weak labels (auto-extracted from reports)
- Multi-label overlap
- Varying image quality
- High noise in lung regions

####  Summary
This dataset is well-suited for **multi-label classification**, but requires:
- Class balancing techniques
- Robust preprocessing
- Careful evaluation metrics (AUROC, F1-score)


---

### Dataset 2: Skin Cancer Dataset (HAM10000)

####  Source
Available on **:contentReference[oaicite:3]{index=3}**

####  Basic Information
- **Type:** Dermoscopic skin lesion images
- **Images:** ~10,000
- **Classes:** 7  
  (Melanoma, Nevus, Basal cell carcinoma, Actinic keratosis, etc.)

####  Dataset Imbalance
- Benign nevus class dominates
- Melanoma samples are significantly fewer

####  Challenges
- High visual similarity between classes
- Skin tone and lighting variation
- Small lesion sizes
- Annotation uncertainty

####  Summary
This dataset is widely used for **skin cancer detection**, but requires:
- Data augmentation
- Color normalization
- Advanced CNN models (ResNet, EfficientNet)



---

##  Conclusion
Week 2 focused on developing a strong foundation in image preprocessing and understanding real-world challenges in medical imaging datasets. These skills are essential for building reliable and clinically useful machine learning models.

---


#  Week 3: Skin Disease Image Classification using YOLOv8

##  Objective
The objective of Week 3 was to explore a medical image dataset from Roboflow and train a deep learning–based **image classification model** using **Ultralytics YOLOv8** on Google Colab with GPU acceleration.

---

##  Dataset Exploration

###  Dataset Source
- **Platform:** Roboflow Universe  
- **Dataset Name:** Skin Disease Dataset  
- **Version:** skin-1  

###  Type of Images
- RGB clinical / dermoscopic skin lesion images
- Images vary in lighting, skin tone, and lesion appearance

###  Classes
The dataset contains **two skin disease classes**:
1. **Basal Cell Carcinoma**
2. **Melanoma**


###  Dataset Splits
- **Training set:** Used to learn model parameters  
- **Validation set:** Used for tuning and performance monitoring  
- **Test set:** Used for final evaluation  

###  Challenges Identified
- Class imbalance (common in medical datasets)
- High visual similarity between lesion types
- Small dataset size → risk of overfitting
- Sensitivity of medical image classification tasks

---

##  Model Training

###  Model Used
- **YOLOv8 Classification Model (Ultralytics)**
- Pretrained weights: `yolov8n-cls.pt`

###  Training Environment
- **Platform:** Google Colab  
- **Hardware:** GPU (CUDA enabled)  
- **Frameworks:** PyTorch, Ultralytics YOLOv8  

###  Training Configuration
- Image size: `224 × 224`
- Optimizer: Adam
- Epochs: 30
- Batch size: 32
- Transfer learning using pretrained weights

###  Training Command
The YOLOv8 model was trained directly using the dataset folder structure provided by Roboflow.

---

##  Model Evaluation & Prediction

###  Validation
- Model performance was monitored using validation accuracy and loss metrics
- Training logs and metrics were stored automatically by Ultralytics

###  Prediction on Test Data
During inference, it was observed that YOLOv8 **does not recursively scan subdirectories** for images.  
To handle this, a wildcard path was used to correctly locate test images.

```python
model.predict(
    source="skin-1/test/*/*",
    save=True
)
```


# Week 4:TIGER TILs Detection using YOLOv8

This Week 4 project demonstrates an end-to-end implementation of a YOLOv8 object detection pipeline to detect and quantify lymphocytes and plasma cells (TILs — Tumor Infiltrating Lymphocytes) in Whole Slide Imaging (WSI) regions of interest (ROIs). The dataset was obtained from Roboflow (wsiroisimages, v1) and formatted for YOLOv8. The model was fine-tuned from a pre-trained YOLOv8s checkpoint and evaluated on the validation split.

## Table of Contents

- [Installation](#installation)
- [Repository structure](#repository-structure)
- [Dataset](#dataset)
  - [Download](#download)
  - [Inspect / Visualize](#inspect--visualize)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Inference](#inference)
- [Results & Expected Outputs](#results--expected-outputs)
- [Reproducibility](#reproducibility)
- [Tips & Troubleshooting](#tips--troubleshooting)
- [Authors / Contact](#authors--contact)
- [License](#license)

---

## Installation

Recommended Python: 3.8+

Install the minimal required packages used for this project:

```bash
pip install -q ultralytics roboflow opencv-python matplotlib
```

Notes:
- Install PyTorch separately according to your platform and CUDA version if you plan to train with GPU acceleration. See https://pytorch.org/.
- ultralytics includes YOLOv8 APIs used in the training examples below.

---

## Repository structure (expected for Week4)

- `Week4/`
  - `notebooks/` — Jupyter notebooks used for experimentation
    - `Week4_experiment.ipynb`
  - `data/` — (optional) raw / processed dataset or download scripts
  - `wsiroisimages-1/` — Roboflow downloaded dataset (YOLO format)
    - `train/images`, `train/labels`, `valid/images`, `valid/labels`
    - `data.yaml`
  - `src/` — optional scripts (train.py / eval.py / predict.py)
  - `results/` — saved models, logs, visual outputs
  - `README.md` — this file

If your folder layout differs, update paths in the examples below accordingly.

---

## Dataset

Dataset: `wsiroisimages` (Roboflow) — version 1, YOLO format. Contains WSI ROI images annotated for lymphocytes and plasma cells.

### Download (Roboflow)

Replace `YOUR_ROBOFLOW_API_KEY` with your Roboflow API key:

```python
from roboflow import Roboflow
import yaml

rf = Roboflow(api_key="YOUR_ROBOFLOW_API_KEY")  # Replace with your Roboflow API key
project = rf.workspace("xray-u9rf3").project("wsiroisimages")
dataset = project.version(1).download("yolov8")

with open("wsiroisimages-1/data.yaml", "r") as f:
    data_cfg = yaml.safe_load(f)
print(data_cfg)
```

### Inspect / Visualize

Quick check to visualize one training image and its YOLO-format bounding boxes:

```python
import os
import cv2
import matplotlib.pyplot as plt

img_path = "wsiroisimages-1/train/images"
lbl_path = "wsiroisimages-1/train/labels"

img_file = os.listdir(img_path)[0]
label_file = img_file.replace(".jpg", ".txt")

img = cv2.imread(os.path.join(img_path, img_file))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

h, w, _ = img.shape

with open(os.path.join(lbl_path, label_file)) as f:
    for line in f:
        cls, xc, yc, bw, bh = map(float, line.split())
        x1 = int((xc - bw/2) * w)
        y1 = int((yc - bh/2) * h)
        x2 = int((xc + bw/2) * w)
        y2 = int((yc + bh/2) * h)
        cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,0), 1)

plt.imshow(img)
plt.axis("off")
plt.show()
```

---

## Model Training

This project fine-tuned a pre-trained YOLOv8s model for 50 epochs with batch size 8 using the AdamW optimizer.

Example training code (Ultralytics YOLO API):

```python
from ultralytics import YOLO

model = YOLO("yolov8s.pt")

model.train(
    data="wsiroisimages-1/data.yaml",
    imgsz=512,
    epochs=50,
    batch=8,
    optimizer="AdamW",
    lr0=1e-3,
    device=0,  # GPU device index; change to 'cpu' or omit if using CPU
    workers=4,
    project="TIGER_TILs",
    name="yolov8_tils_detector"
)
```

Hyperparameters used (example):
- img size: 512
- epochs: 50
- batch size: 8
- optimizer: AdamW
- initial learning rate: 1e-3

Adjust these depending on GPU memory and dataset size.

---

## Evaluation

After training, validate the model on the validation set to compute standard detection metrics (Precision, Recall, mAP50, mAP50-95).

Example:

```python
metrics = model.val()
print(metrics)
```

Typical evaluation outputs:
- Precision, Recall
- mAP@0.50
- mAP@0.50:0.95 (COCO-style)
- Per-class metrics (if multi-class)

---

## Inference

Run inference on the validation images (or any image/directory). Results (image outputs with predictions) can be saved.

Example:

```python
model.predict(
    source="wsiroisimages-1/valid/images",
    imgsz=512,
    conf=0.25,
    save=True
)
```

Outputs:
- Annotated images saved to `runs/detect/predict` (Ultralytics default) or under the `project/name` folder if specified.

You can also perform single-image inference and programmatically parse predictions for quantification.

---


