import streamlit as st
import pandas as pd
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import re
import json
from datetime import datetime
import sqlite3
import spacy
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
import warnings
warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="Medical Report PDF Extractor",
    page_icon="🏥",
    layout="wide"
)

# Initialize session state
if 'extracted_data' not in st.session_state:
    st.session_state.extracted_data = []

class MedicalReportExtractor:
    def __init__(self):
        self.setup_models()
        self.setup_database()
    
    def setup_models(self):
        """Initialize NLP models for medical text processing"""
        try:
            # Load spaCy model for general NLP
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                st.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
                self.nlp = None
            
            # Medical NER model (using BioBERT-based model)
            try:
                self.medical_ner = pipeline(
                    "ner",
                    model="d4data/biomedical-ner-all",
                    tokenizer="d4data/biomedical-ner-all",
                    aggregation_strategy="simple"
                )
            except Exception as e:
                st.warning(f"Medical NER model not available: {str(e)}")
                self.medical_ner = None
                
        except Exception as e:
            st.error(f"Error setting up models: {str(e)}")
    
    def setup_database(self):
        """Setup SQLite database for storing extracted data"""
        self.conn = sqlite3.connect(':memory:', check_same_thread=False)
        cursor = self.conn.cursor()
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS medical_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            patient_name TEXT,
            patient_id TEXT,
            age TEXT,
            sex TEXT,
            date_of_report DATE,
            doctor_name TEXT,
            hospital_name TEXT,
            study_type TEXT,
            diagnosis TEXT,
            clinical_findings TEXT,
            medications TEXT,
            lab_values TEXT,
            vital_signs TEXT,
            extracted_text TEXT,
            extraction_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        self.conn.commit()
    
    def extract_text_from_pdf(self, pdf_file):
        """Extract text from PDF using multiple methods"""
        text_content = ""
        
        try:
            # Method 1: Direct text extraction using PyMuPDF
            pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
            
            for page_num in range(len(pdf_document)):
                page = pdf_document.load_page(page_num)
                text_content += page.get_text()
                
                # Method 2: OCR for images/scanned PDFs
                if len(text_content.strip()) < 100:  # Likely scanned PDF
                    pix = page.get_pixmap()
                    img_data = pix.tobytes("png")
                    img = Image.open(io.BytesIO(img_data))
                    ocr_text = pytesseract.image_to_string(img)
                    text_content += ocr_text
            
            pdf_document.close()
            return text_content
            
        except Exception as e:
            st.error(f"Error extracting text from PDF: {str(e)}")
            return ""
    
    def extract_medical_entities(self, text):
        """Extract medical entities using NER models"""
        entities = {
            'medications': [],
            'conditions': [],
            'lab_values': [],
            'procedures': []
        }
        
        if self.medical_ner:
            try:
                ner_results = self.medical_ner(text)
                for entity in ner_results:
                    label = entity['entity_group'].lower()
                    if 'drug' in label or 'medication' in label:
                        entities['medications'].append(entity['word'])
                    elif 'disease' in label or 'condition' in label:
                        entities['conditions'].append(entity['word'])
                    elif 'procedure' in label:
                        entities['procedures'].append(entity['word'])
            except Exception as e:
                st.warning(f"NER extraction failed: {str(e)}")
        
        return entities
    
    def extract_structured_data(self, text):
        """Extract structured data using regex patterns"""
        data = {}
        
        # Patient name patterns (enhanced for medical reports)
        name_patterns = [
            r'Patient\s*Name\s*:?\s*([A-Za-z\s\.]+?)(?:\s+Study\s+Date|\s+Age|\n)',
            r'Name\s*:?\s*([A-Za-z\s\.]+?)(?:\s+Study\s+Date|\s+Age|\n)',
            r'Patient\s*:?\s*([A-Za-z\s\.]+?)(?:\s+Study\s+Date|\s+Age|\n)',
            r'Mr\.?\s*([A-Za-z\s\.]+)',
            r'Mrs\.?\s*([A-Za-z\s\.]+)',
            r'Ms\.?\s*([A-Za-z\s\.]+)',
            r'Dr\.?\s*([A-Z\s\.]+?)(?:\s+Study\s+Date|\s+Age|\n)'
        ]
        
        for pattern in name_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                data['patient_name'] = match.group(1).strip()
                break
        
        # Patient ID patterns (enhanced)
        id_patterns = [
            r'UHID\s*:?\s*([A-Z0-9\.]+)',
            r'Patient\s*ID\s*:?\s*([A-Z0-9\.]+)',
            r'ID\s*:?\s*([A-Z0-9\.]+)',
            r'Registration\s*No\s*:?\s*([A-Z0-9\.]+)',
            r'IPD\s*No\s*:?\s*([A-Z0-9\.]+)'
        ]
        
        for pattern in id_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                data['patient_id'] = match.group(1).strip()
                break
        
        # Date patterns (enhanced for various formats)
        date_patterns = [
            r'Study\s*Date\s*:?\s*(\d{1,2}[-/]\w{3}[-/]\d{2,4})',
            r'Date\s*:?\s*(\d{1,2}[-/]\w{3}[-/]\d{2,4})',
            r'Study\s*Date\s*:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            r'Date\s*:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            r'Date\s*of\s*Report\s*:?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            r'(\d{1,2}[-/]\w{3}[-/]\d{2,4})'
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                data['date_of_report'] = match.group(1).strip()
                break
        
        # Age and Sex extraction
        age_patterns = [
            r'Age\s*:?\s*(\d+)\s*Years?',
            r'Age\s*:?\s*(\d+)\s*Y',
            r'Age\s*:?\s*(\d+)'
        ]
        
        for pattern in age_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                data['age'] = match.group(1).strip()
                break
        
        sex_patterns = [
            r'Sex\s*:?\s*([MF])',
            r'Gender\s*:?\s*([MF])',
            r'Sex\s*:?\s*(Male|Female)'
        ]
        
        for pattern in sex_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                data['sex'] = match.group(1).strip()
                break
        
        # Doctor name patterns (enhanced)
        doctor_patterns = [
            r'Referring\s*Doctor\s*:?\s*([A-Za-z\s\.]+?)(?:\s+Age|\n)',
            r'Dr\.?\s*([A-Za-z\s\.]+?)(?:\s+Study|\s+Age|\n)',
            r'Doctor\s*:?\s*([A-Za-z\s\.]+)',
            r'Physician\s*:?\s*([A-Za-z\s\.]+)',
            r'Radiologist\s*:?\s*([A-Za-z\s\.]+)'
        ]
        
        for pattern in doctor_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                data['doctor_name'] = match.group(1).strip()
                break
        
        # Hospital/Clinic name (enhanced)
        hospital_patterns = [
            r'Hospital\s*:?\s*([A-Za-z\s\.]+)',
            r'Clinic\s*:?\s*([A-Za-z\s\.]+)',
            r'Medical\s*Center\s*:?\s*([A-Za-z\s\.]+)',
            r'Dept\.\s*No\.\s*:?\s*([A-Za-z0-9\s\.]+)'
        ]
        
        for pattern in hospital_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                data['hospital_name'] = match.group(1).strip()
                break
        
        # Study/Examination type
        study_patterns = [
            r'(CHEST)\s*(?:X-?RAY)?',
            r'(CT)\s*(?:SCAN)?',
            r'(MRI)',
            r'(ULTRASOUND)',
            r'Study\s*Type\s*:?\s*([A-Za-z\s]+)',
            r'Examination\s*:?\s*([A-Za-z\s]+)'
        ]
        
        for pattern in study_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                data['study_type'] = match.group(1).strip()
                break
        
        # Clinical findings (enhanced for radiology reports)
        findings_patterns = [
            r'(?:CHEST|Findings?|Impression|Report)\s*:?\s*(.*?)(?=\n\n|\Z)',
            r'Clinical\s*Findings?\s*:?\s*(.*?)(?=\n\n|\Z)',
            r'Radiological\s*Findings?\s*:?\s*(.*?)(?=\n\n|\Z)'
        ]
        
        findings_text = ""
        for pattern in findings_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                findings_text = match.group(1).strip()
                break
        
        # If no specific findings section, extract key medical observations
        if not findings_text:
            # Extract lines that contain medical terms
            medical_keywords = [
                'opacity', 'pneumonia', 'consolidation', 'effusion', 
                'nodule', 'mass', 'infiltrate', 'atelectasis',
                'heart', 'cardiac', 'aortic', 'diaphragm', 'thorax'
            ]
            
            lines = text.split('\n')
            medical_lines = []
            for line in lines:
                if any(keyword.lower() in line.lower() for keyword in medical_keywords):
                    medical_lines.append(line.strip())
            
            findings_text = '\n'.join(medical_lines)
        
        data['clinical_findings'] = findings_text
        
        # Diagnosis patterns (enhanced)
        diagnosis_patterns = [
            r'(?:pneumonia|Pneumonia)',
            r'(?:consolidation|Consolidation)',
            r'(?:effusion|Effusion)',
            r'(?:normal|Normal)',
            r'Diagnosis\s*:?\s*([A-Za-z0-9\s,.-]+?)(?:\n|\r|$)',
            r'Impression\s*:?\s*([A-Za-z0-9\s,.-]+?)(?:\n|\r|$)'
        ]
        
        diagnoses = []
        for pattern in diagnosis_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            diagnoses.extend(matches)
        
        # Extract specific conditions mentioned
        if 'pneumonia' in text.lower():
            diagnoses.append('Pneumonia')
        if 'normal' in text.lower() and ('heart' in text.lower() or 'cardiac' in text.lower()):
            diagnoses.append('Normal cardiac silhouette')
        
        data['diagnosis'] = '; '.join(set(diagnoses)) if diagnoses else ''
        
        # Lab values (basic pattern)
        lab_values = []
        lab_patterns = [
            r'(\w+)\s*:?\s*(\d+\.?\d*)\s*(mg/dl|mmol/L|g/dl|%|/μL)',
            r'(\w+)\s*-?\s*(\d+\.?\d*)\s*(mg/dl|mmol/L|g/dl|%|/μL)'
        ]
        
        for pattern in lab_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                lab_values.append(f"{match[0]}: {match[1]} {match[2]}")
        
        data['lab_values'] = '; '.join(lab_values) if lab_values else ''
        
        # Vital signs
        vital_patterns = [
            r'BP\s*:?\s*(\d+/\d+)',
            r'Blood\s*Pressure\s*:?\s*(\d+/\d+)',
            r'Pulse\s*:?\s*(\d+)',
            r'Temperature\s*:?\s*(\d+\.?\d*)'
        ]
        
        vitals = []
        for pattern in vital_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                vitals.append(match.group(0))
        
        data['vital_signs'] = '; '.join(vitals) if vitals else ''
        
        return data
    
    def save_to_database(self, filename, extracted_data, full_text):
        """Save extracted data to database"""
        cursor = self.conn.cursor()
        
        cursor.execute('''
        INSERT INTO medical_reports 
        (filename, patient_name, patient_id, age, sex, date_of_report, doctor_name, 
         hospital_name, study_type, diagnosis, clinical_findings, medications, lab_values, vital_signs, extracted_text)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            filename,
            extracted_data.get('patient_name', ''),
            extracted_data.get('patient_id', ''),
            extracted_data.get('age', ''),
            extracted_data.get('sex', ''),
            extracted_data.get('date_of_report', ''),
            extracted_data.get('doctor_name', ''),
            extracted_data.get('hospital_name', ''),
            extracted_data.get('study_type', ''),
            extracted_data.get('diagnosis', ''),
            extracted_data.get('clinical_findings', ''),
            extracted_data.get('medications', ''),
            extracted_data.get('lab_values', ''),
            extracted_data.get('vital_signs', ''),
            full_text[:2000]  # Store first 2000 chars
        ))
        
        self.conn.commit()
        return cursor.lastrowid
    
    def get_all_records(self):
        """Retrieve all records from database"""
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM medical_reports ORDER BY extraction_timestamp DESC')
        columns = [description[0] for description in cursor.description]
        rows = cursor.fetchall()
        return pd.DataFrame(rows, columns=columns)

def main():
    st.title("🏥 Medical Report PDF Data Extractor")
    st.markdown("Extract structured data from medical reports and store in database")
    
    # Initialize extractor
    if 'extractor' not in st.session_state:
        with st.spinner("Initializing models..."):
            st.session_state.extractor = MedicalReportExtractor()
    
    extractor = st.session_state.extractor
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # OCR settings
        st.subheader("OCR Settings")
        tesseract_config = st.text_input(
            "Tesseract Config",
            value="--psm 6 -l eng",
            help="Tesseract OCR configuration"
        )
        
        # Database operations
        st.subheader("Database Operations")
        if st.button("Clear Database"):
            cursor = extractor.conn.cursor()
            cursor.execute("DELETE FROM medical_reports")
            extractor.conn.commit()
            st.success("Database cleared!")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📄 Upload PDF")
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload one or more medical report PDF files"
        )
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                st.write(f"**File:** {uploaded_file.name}")
                
                if st.button(f"Extract Data from {uploaded_file.name}", key=uploaded_file.name):
                    with st.spinner(f"Processing {uploaded_file.name}..."):
                        # Extract text
                        text_content = extractor.extract_text_from_pdf(uploaded_file)
                        
                        if text_content:
                            # Extract structured data
                            structured_data = extractor.extract_structured_data(text_content)
                            
                            # Extract medical entities
                            medical_entities = extractor.extract_medical_entities(text_content)
                            
                            # Combine medications from entities
                            if medical_entities['medications']:
                                structured_data['medications'] = '; '.join(medical_entities['medications'])
                            
                            # Save to database
                            record_id = extractor.save_to_database(
                                uploaded_file.name, 
                                structured_data, 
                                text_content
                            )
                            
                            st.success(f"Data extracted and saved! Record ID: {record_id}")
                            
                            # Display extracted data
                            st.subheader("Extracted Data")
                            for key, value in structured_data.items():
                                if value:
                                    st.write(f"**{key.replace('_', ' ').title()}:** {value}")
                            
                            # Show medical entities
                            if any(medical_entities.values()):
                                st.subheader("Medical Entities")
                                for entity_type, entities in medical_entities.items():
                                    if entities:
                                        st.write(f"**{entity_type.title()}:** {', '.join(entities)}")
                        else:
                            st.error("Failed to extract text from PDF")
    
    with col2:
        st.header("🗄️ Database Records")
        
        # Display all records
        df = extractor.get_all_records()
        
        if not df.empty:
            st.write(f"**Total Records:** {len(df)}")
            
            # Display summary statistics
            with st.expander("📊 Summary Statistics"):
                st.write(f"- **Unique Patients:** {df['patient_name'].nunique()}")
                st.write(f"- **Unique Doctors:** {df['doctor_name'].nunique()}")
                st.write(f"- **Unique Hospitals:** {df['hospital_name'].nunique()}")
            
            # Search functionality
            search_term = st.text_input("🔍 Search records", placeholder="Enter patient name, doctor, etc.")
            
            if search_term:
                mask = df.astype(str).apply(lambda x: x.str.contains(search_term, case=False, na=False)).any(axis=1)
                filtered_df = df[mask]
            else:
                filtered_df = df
            
            # Display records
            if not filtered_df.empty:
                # Select columns to display
                display_columns = st.multiselect(
                    "Select columns to display",
                    options=df.columns.tolist(),
                    default=['filename', 'patient_name', 'age', 'sex', 'study_type', 'diagnosis', 'extraction_timestamp']
                )
                
                if display_columns:
                    st.dataframe(
                        filtered_df[display_columns],
                        use_container_width=True,
                        height=400
                    )
                
                # Export functionality
                st.subheader("📤 Export Data")
                col_exp1, col_exp2 = st.columns(2)
                
                with col_exp1:
                    if st.button("Export to CSV"):
                        csv = filtered_df.to_csv(index=False)
                        st.download_button(
                            label="Download CSV",
                            data=csv,
                            file_name=f"medical_reports_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                
                with col_exp2:
                    if st.button("Export to Excel"):
                        output = io.BytesIO()
                        with pd.ExcelWriter(output, engine='openpyxl') as writer:
                            filtered_df.to_excel(writer, sheet_name='Medical Reports', index=False)
                        
                        st.download_button(
                            label="Download Excel",
                            data=output.getvalue(),
                            file_name=f"medical_reports_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
            else:
                st.info("No records found matching your search criteria.")
        else:
            st.info("No records in database. Upload and process some PDF files to get started!")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**Note:** This tool uses OCR and NLP models to extract data from medical reports. "
        "Always verify the extracted information for accuracy and completeness."
    )

if __name__ == "__main__":
    main()