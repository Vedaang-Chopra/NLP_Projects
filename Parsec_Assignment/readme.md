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
---

## 🗂️ Data Description

### Complete Data
- **Sheet Name:** `DataSet.xlsx`  
- **Columns:**
  - `datasheet_link` – URL to the hosted PDF.
  - `target_col` – The target category (Lighting, Fuses, Cables, Others).  

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
  - **Class Probabilities**  - > Shared the confidence of the majority class

### 4. Model Evaluation and Testing
- Make predictions on the test dataset.  
- Report performance metrics like **Accuracy**, **F1-Score**, and **Confusion Matrix**.

---

## 📦 Deliverables

1. **Code**  
   - `build_dataset_v2.ipynb` – Implements the data collection and PDF extraction to build the raw dataset
   - `eda_cleaning.ipynb` – Implements the exploratory data analysis, text preprocessing and splitting data to training and testing.  
   - `ml_modelling.ipynb` – Implemented bag of words technique and used ML models to perform classification.
   - `dl_modelling.ipynb` – Implemented Glove Embedding and LSTM based classification technique to perform classification.
   - `inference.ipynb` – Contains the inference pipeline to classify PDFs from URLs.  
   
2. **Inference Pipeline**  
   - The inference logic is implemented in `inference.ipynb`.

3. **Documentation**  
   - Detailed walkthrough of the solution, including architecture and model choice.  
   - Time taken to complete the problem: **6-9 hours**.  
   - **Model Chosen:** Multiple like TF-IDF with Random Forest or Glove embedding with LSTM for simplicity and interpretability. Used text classification by extracting text from PDF and classifying it.
   - **Shortcomings:**  
     - Excluded HTML files, processed only PDFs.  
     - Requires further review to ensure no overlooked data.  
     - PDF extraction might need refinement for edge cases.  
     - Needs to be tested in a real production environment.

4. **Model Performance Report**  
   - Since this is a classification problem on a balanced dataset, using accuracy, precision, recall, f-1 score and confusion matrix, can help us thourgoughly understand the workings of the model trained.
   - **Test Accuracy:** 0.95  
   - **Classification Report:**
     ```
                 precision    recall  f1-score   support

           0       0.97      0.98      0.97        90
           1       0.97      1.00      0.99       113
           2       0.94      0.84      0.89        61
           3       0.89      0.92      0.90        85
      ```
    
    ```
    
    accuracy                               0.95       349
    macro avg          0.94      0.93      0.94       349
    weighted avg       0.95      0.95      0.94       349
    ```


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
    python  Run the ml_modelling or dl_modelling.ipynb
```

## 3. Inference
```bash
python inference.ipynb
```
---


