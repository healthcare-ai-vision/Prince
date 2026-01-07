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

