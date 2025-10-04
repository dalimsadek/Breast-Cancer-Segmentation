# 🩺 Breast Cancer Detection using Deep Learning and Transfer Learning

## 📘 Overview
This project focuses on **automated breast cancer detection** and **tumor segmentation** from mammogram images using **deep learning**.  
It integrates **U-Net** and **VGG16-based models** with **transfer learning** and **ensemble techniques** to improve classification and segmentation accuracy.

---

## 🧠 Project Structure

```
├── Csv_files/                     # Training and testing CSVs
│   ├── calc_case_description_train_set.csv
│   ├── calc_case_description_test_set.csv
│   ├── mass_case_description_train_set.csv
│   └── mass_case_description_test_set.csv
│
├── Preprocessing/                 # Data preprocessing and preparation scripts
│   ├── Exploring_the_data.ipynb
│   ├── Merge_multi_tumor.py
│   ├── Restructuring_the_data.py
│   ├── Update_Paths.py
│   ├── classifying_the_patients.py
│   ├── configuration.py
│   └── data_preprocessing.py
│
├── Training/                      # Model training modules
│   ├── UNET/
│   │   ├── UNET_31_epochs.ipynb
│   │   ├── UNET_51_epochs.ipynb
│   │   └── ensemble_learning.ipynb
│   ├── VGG16/
│   └── working/
│       └── ensemble_learning.ipynb
│
├── Testing/
│   └── plot.py                    # Visual evaluation and metric plotting
│
└── README.md
```

---

## 🧩 Methodology

### 1. **Data Preprocessing**
- Loaded and merged DDSM (Digital Database for Screening Mammography) dataset CSVs.  
- Standardized image paths, labels, and data structures.  
- Implemented augmentation, resizing, and normalization pipelines.

### 2. **Model Development**
- **U-Net Architecture** for **tumor segmentation**:
  - Trained for 31 and 51 epochs for performance comparison.
  - Used binary cross-entropy and Dice coefficient metrics.
- **VGG16 Transfer Learning** for **mass/calcification classification**:
  - Pretrained weights on ImageNet.
  - Fine-tuned for medical imaging domain.

### 3. **Ensemble Learning**
- Combined predictions from multiple U-Net and VGG16 models to enhance stability and robustness.

### 4. **Evaluation**
- Accuracy, Precision, Recall, F1-score, and Dice coefficient used.
- Visual comparisons between ground truth masks and predictions.

---

## 📈 Results
- **U-Net** achieved high segmentation performance on mammogram regions.
- **VGG16** classification accuracy: **≈92%** on test data.
- Ensemble learning improved model generalization and reduced overfitting.

---

## ⚙️ Technologies Used
- **Python 3.9+**
- **TensorFlow / Keras**
- **OpenCV**
- **NumPy / Pandas**
- **Matplotlib / Seaborn**

---

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/Breast-Cancer-Detection.git
   cd Breast-Cancer-Detection
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run preprocessing:
   ```bash
   python Preprocessing/data_preprocessing.py
   ```

4. Train model:
   - Open `Training/UNET/UNET_31_epochs.ipynb` or `Training/UNET/UNET_51_epochs.ipynb`.
   - Run all cells sequentially.

5. Evaluate and visualize:
   ```bash
   python Testing/plot.py
   ```

---

## 📂 Dataset
The project uses the **DDSM (Digital Database for Screening Mammography)** dataset, which includes:
- **Calcification and mass cases** with ROI masks.
- Metadata stored in CSV files for each subset.

> Note: The dataset is publicly available for research purposes.

---

## 👨‍💻 Author
**Mohamed Ali Msadek**  
Data Science Engineer | SUP’COM  
🔗 [LinkedIn](https://www.linkedin.com/in/mohamed-alimsadek)  
📧 mohamedali.msadek02@outlook.fr  
📁 [GitHub](https://github.com/dalimsadek)

---

## 🧾 License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
