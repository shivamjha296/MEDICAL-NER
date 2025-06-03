# Medical Text Extraction System

This application uses Bio_ClinicalBERT and other NLP tools to extract and analyze medical information from PDF documents.

## Features

- PDF text extraction
- Medical entity recognition using Bio_ClinicalBERT
- Basic NER using spaCy
- Interactive Streamlit interface

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
2. Upload a PDF file containing medical text
3. The application will:
   - Extract text from the PDF
   - Identify medical entities
   - Process the text using Bio_ClinicalBERT
   - Display the results in an interactive interface

## Requirements

- Python 3.8+
- See requirements.txt for package dependencies

## Note

The first time you run the application, it will download the Bio_ClinicalBERT model, which might take a few minutes depending on your internet connection. 