# PDF Classification for Building Construction Products

## Project: Machine Learning Engineer 2 Assignment

**Company:** Parspec, Inc.  
**Objective:** Classify product PDFs into one of four categories: [Lighting, Fuses, Cables, Others].

---

## 📖 Overview
Parspec is revolutionizing the sale of building construction products by digitizing and organizing product data globally. This project focuses on developing a machine learning model to classify product PDFs into one of four categories, enhancing product understanding and streamlining the sales process.

---

## 🎯 Problem Statement
The goal is to classify PDF documents into the following categories:
- **Lighting**  
- **Fuses**  
- **Cables**  
- **Others**  

You are provided with URLs linking to PDFs and their corresponding categories, which will serve as training and test data.

---


### **Features of the README:**
- Clearly outlines **project goals, data, and deliverables**.  
- Provides **step-by-step instructions** on running and testing the project.  
- Structured with appropriate **markdown headings and sections** for readability.  


## 🗂️ Data Description

### Complete Data
- **Sheet Name:** `DataSet.xlsx`  
- **Columns:**
  - `datasheet_link` – URL to the hosted PDF.
  - `target_col` – The target category (Lighting, Fuses, Cables, Others).  

<!-- ### Test Data
- **Sheet Name:** `test_data`  
- **Columns:**
  - `datasheet_link` – URL to the hosted PDF.
  - `target_col` – The target category of the PDF. -->

---

## 🛠️ Tasks Breakdown

### 1. PDF Text Extraction Pipeline
- Develop a pipeline to download and extract text from PDFs.  
- Use tools like **PyMuPDF (fitz)** for text extraction and **Tesseract OCR** for scanned PDFs.

### 2. Model Development
- Train a machine learning model to classify the extracted PDF text into one of the four categories.  
- Models to explore:
  - **LSTM (PyTorch)**  
  - **TF-IDF + Logistic Regression**  
  - **TF-IDF + Random Forest**  

### 3. Inference Pipeline
- Create an inference function that accepts a PDF URL as input and returns:  
  - **Predicted Class** – One of [Lighting, Fuses, Cables, Others].  
  - **Class Probabilities**  

### 4. Model Evaluation and Testing
- Make predictions on the test dataset.  
- Report performance metrics like **Accuracy**, **F1-Score**, and **Confusion Matrix**.

---

## 📦 Deliverables

1. **Code** – Full codebase for PDF extraction, model training, and inference.  
2. **Inference Pipeline** – A Python function or hosted app for classifying PDFs from URLs.  
3. **Documentation**  
   - Solution walkthrough (Explain architecture, models, and performance).  
   - Time taken to solve the problem.  
   - Model choice explanation.  
   - Shortcomings and potential improvements.  
4. **Model Performance Report**  
   - Performance metrics (Accuracy, Precision, Recall, F1-Score).  
   - Justification for the choice of metrics.

---

## 🚀 How to Run the Project

### 1. Environment Setup
```bash
# Create virtual environment
conda create -n parspec_pdf_classification python=3.11
conda activate parspec_pdf_classification

# Install dependencies
pip install -r requirements.txtx
```

## 2. Training the Model
```bash

python train_model.py
```


## 3. Inference
```bash
python inference.py --url https://example.com/sample.pdf
```
---


