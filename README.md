# 🧠 Cassiopeia Project - 3D Brain Tumor Segmentation with Mamba

## 📌 Context  
This project was carried out as part of the **Cassiopée program** at **Télécom SudParis**, under the supervision of **Prof. Nicolas Rougon**, in collaboration with **Gustavo Paulino** and **Mariana Meirelles**.  

Our work received a **double award**:  
- 🏆 **Master Award** at the *Évry-Sénart Sciences and Innovation Conference* (dedicated to AI and Healthcare).  
- 🥈 **2nd highest grade** among the **108 Cassiopée projects** presented this year.  

---

## 🎯 Objective  
The main objective of this project was to explore **Mamba neural architectures** applied to **3D medical image segmentation** (MRI scans from the **BraTS2021 dataset**).  

Specifically, we:  
- Studied and improved the **SegMamba** and **UMamba** models.  
- Proposed a new architecture, **VM-Unet3D**, inspired by VM-Unet and adapted for 3D data.  
- Integrated all models into the **nnU-Net framework** for standardized training and evaluation.  
- Developed a **Streamlit web interface** to test and visualize the models.  

---

## ⚙️ Streamlit Interface Features  
- Upload of medical images (**NIFTI, DICOM, etc.**)  
- Automatic segmentation using **SegMamba**【segmamba.py】  
- Multi-view visualization (axial, coronal, sagittal) with colored overlays【viewer.py】  
- Tumor statistics calculation (volume, maximum diameter, etc.)  
- Export of an automatic **PDF report** with segmentations and metrics【app.py】  

---
