# Medical Form Extraction System

This application uses advanced NLP models to extract and analyze medical information from medical forms and documents.

## Features

- PDF text extraction with layout preservation using pdfplumber
- Medical entity recognition using biomedical NER models
- Structured form field extraction
- Medical condition screening results extraction
- Interactive Streamlit interface with detailed statistics

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Download the spaCy model:
```bash
python -m spacy download en_core_web_sm
```

## Running the Application

To run the application, execute:
```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501` in your web browser.

## Usage

1. Open the application in your web browser
2. Upload a medical form PDF file
3. The application will:
   - Extract and clean text while preserving layout
   - Identify and extract form fields (MC Ref, Name, Age, etc.)
   - Extract medical conditions and their results
   - Process the text using medical NER models
   - Display structured results with categories and values
   - Show extraction statistics

## Features

- **Form Field Extraction:**
  - Patient Information (Name, Age, Passport)
  - Medical Reference Numbers
  - Dates and Administrative Data

- **Medical Condition Screening:**
  - HIV/AIDS
  - TB
  - Malaria
  - Leprosy
  - Hepatitis
  - Urine Test Results
  - Fitness Status

- **Structured Output:**
  - Categorized entities
  - Field-value pairs
  - Medical conditions with results
  - Extraction statistics

## Requirements

- Python 3.8+
- See requirements.txt for package dependencies

## Note

The first time you run the application, it will download the medical NER model, which might take a few minutes depending on your internet connection. 