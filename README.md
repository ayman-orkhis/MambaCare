# 🧠 Cassiopeia — 3D Brain Tumor Segmentation with Mamba

[![Streamlit](https://img.shields.io/badge/Streamlit-app-red)](#-run-the-streamlit-app)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)]()

## 📌 Context
This project was carried out as part of the **Cassiopée program** at **Télécom SudParis**, under the supervision of **Prof. Nicolas Rougon**, in collaboration with **Gustavo Paulino** and **Mariana Meirelles**.

Our work received a **double award**:
- 🏆 **Master Award** at the *Évry-Sénart Sciences & Innovation Conference* (AI & Healthcare).
- 🥈 **2nd highest grade** among the **108 Cassiopée projects** presented this year.

---

## 🎯 Goal
Explore and evaluate **Mamba-based neural architectures** for **3D brain tumor segmentation** on MRI (BraTS2021), benchmarked against CNN/Transformer baselines and exposed through an easy-to-use Streamlit app.

**What we did**
- Studied and improved **SegMamba** and **UMamba**.
- Proposed a new model **VM-Unet3D** (inspired by VM-Unet, adapted for 3D).
- Integrated all models into the **nnU-Net** framework for consistent preprocessing, training, and evaluation.
- Built a **Streamlit web interface** for inference, visualization, metrics, and PDF reporting.

---

## ✨ Features (App)
- Upload medical images (**NIfTI** `.nii` / `.nii.gz`; other formats planned).
- One-click segmentation with **SegMamba**.
- Tri-planar views (axial / coronal / sagittal) with **BraTS color overlay**.
- Tumor metrics: volume (cm³), max diameter (mm), extents.
- **Automatic PDF report** with snapshots + measurements.

---

## 🧱 Architecture (Short)
- **SegMamba** encoder with 3D Mamba blocks (+ GSC & channel-MLP), UNETR-style decoder (MONAI), optional **deep supervision**.
- Training & evaluation via **nnU-Net** (v1 for SegMamba; v2 for UMamba / VM-Unet variants).
- App built with **Streamlit**, visualization utilities with **PIL / NumPy**.

---

## 📦 Project Structure
- Other files are not shown in the github for privacy reasons
```
├── app.py                   # Streamlit interface (loading, inference, views, PDF export)
├── segmamba.py              # SegMamba model (encoder/decoder blocks, deep supervision)
├── viewer.py                # Overlay utilities (BraTS color map, tri-planar rendering)
├── FinalReportCassiope.pdf  # Project report (methodology, models, results)
└── requirements.txt         # Python dependencies
```

---

## 🚀 Installation

### 1) Clone
```bash
git clone https://github.com/ayman-orkhis/MambaCare.git
cd MambaCare
```

### 2) Create env & install
```bash
python -m venv venv
# Linux/Mac
source venv/bin/activate
# Windows
# venv\Scripts\activate

pip install -r requirements.txt
```

> Tip: Ensure **Python 3.10+**. A GPU is recommended for real model inference but not mandatory to try the UI (the app includes a safe fallback).

---

## ▶️ Run the Streamlit App
```bash
streamlit run app.py
```
Open the local URL shown in the terminal (typically http://localhost:8501), upload a **.nii / .nii.gz** MRI, run segmentation, inspect tri-planar views, and export the **PDF report**.

---

## 🧪 Dataset (for training)
- **BraTS2021** multiparametric MRI (T1, T1-CE, T2, FLAIR) with expert annotations.
- We use **nnU-Net** preprocessing (normalization, resampling, cropping, patches) and its standard directory structure for tasks.
  
---

## 🏋️ Training (Optional)
Training was conducted via **nnU-Net**:

1. Organize BraTS2021 following `nnUNet_raw/`, run:
   - `nnUNet_plan_and_preprocess`
2. Train (2D/3D full-res as needed) with:
   - `nnUNet_train`
3. Evaluate:
   - `nnUNet_evaluate_folder`
4. Integrate custom trainers (SegMamba / UMamba / VM-Unet3D) for apples-to-apples comparison.

> This repo focuses on **inference & demo**. For full training pipelines, adapt your nnU-Net setup to these architectures.

---

## 🧰 Tech Stack
- **Python**, **PyTorch**, **MONAI**, **nnU-Net**
- **Streamlit**, **NumPy**, **Pillow (PIL)**, **SimpleITK / nibabel**
- **reportlab** for generating PDF reports
---

## 🙏 Acknowledgements
- **Prof. Nicolas Rougon** (supervision)
- **Gustavo Paulino** & **Mariana Meirelles** (teamwork & contributions)
- **BraTS organizers** and **nnU-Net / MONAI and Segmamba authors** communities

