import streamlit as st
import PyPDF2
import torch
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import spacy
from io import BytesIO

# Set page config
st.set_page_config(
    page_title="Medical Text Extraction",
    page_icon="🏥",
    layout="wide"
)

# Initialize session state
if 'extracted_text' not in st.session_state:
    st.session_state.extracted_text = None
if 'entities' not in st.session_state:
    st.session_state.entities = None

# Load Bio_ClinicalBERT model
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    model = AutoModel.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
    return tokenizer, model

# Load spaCy model for basic NER
@st.cache_resource
def load_spacy():
    return spacy.load("en_core_web_sm")

def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def process_text_with_bert(text, tokenizer, model):
    # Tokenize the text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    
    # Get model outputs
    with torch.no_grad():
        outputs = model(**inputs)
    
    return outputs

def main():
    st.title("🏥 Medical Text Extraction System")
    st.write("Upload a medical document to extract and analyze its content")

    # Load models
    try:
        tokenizer, model = load_model()
        nlp = load_spacy()
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return

    # File uploader
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        # Extract text from PDF
        try:
            text = extract_text_from_pdf(uploaded_file)
            st.session_state.extracted_text = text
            
            # Display extracted text
            with st.expander("View Extracted Text"):
                st.text_area("Text Content", text, height=300)
            
            # Process with spaCy for basic NER
            doc = nlp(text)
            entities = [(ent.text, ent.label_) for ent in doc.ents]
            st.session_state.entities = entities
            
            # Display entities
            st.subheader("Identified Entities")
            if entities:
                df = pd.DataFrame(entities, columns=['Entity', 'Type'])
                st.dataframe(df)
            else:
                st.info("No entities found in the text")
            
            # Process with Bio_ClinicalBERT
            st.subheader("Bio_ClinicalBERT Analysis")
            with st.spinner("Processing with Bio_ClinicalBERT..."):
                bert_outputs = process_text_with_bert(text, tokenizer, model)
                st.success("Processing complete!")
                
                # Display some basic statistics
                st.write(f"Text length: {len(text)} characters")
                st.write(f"Number of entities found: {len(entities)}")
                
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")

if __name__ == "__main__":
    main() 