# app.py
# EHR Assistant
"""
### Outcomes AI Case Study - EHR Assistant ###

** Objective: Chatbot to support clinicians to **help find/review patient info.** in EHR via natural language queries

** Benefit: Improves productivity by reducing considerable time spent in reviweing patient details before each appointments

** Overview: This is a Streamlit app that allows clinicians to ask questions about a specific patient's medical records using LangChain and OpenAI's GPT-4 model.
It uses a vector store to index the patient's EHR data and retrieve relevant information based on the user's query.
Converts nested EHR list-of-dicts structure into flat text documents.
Author: Ishwariya
"""

import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers import MergerRetriever
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from langchain.prompts import PromptTemplate
import json
import fitz  # PyMuPDF
import os
from dotenv import load_dotenv
st.set_page_config(page_title="EHR Clinician Chatbot", layout="wide")

load_dotenv()

# Set your OpenAI API key
OPENAI_API_KEY = "OPENAI_API_KEY"

## Helper Functions:


# Load and flatten EHR JSON for a specific patient
def flatten_ehr_json(ehr_data):
    """
    Converts nested EHR list-of-dicts structure into flat text documents.
    Returns a list of langchain Document objects.
    """
    docs = []
    for section in ehr_data:
        for section_name, section_data in section.items():
            # create string representation of the section
            section_text = f"Section: {section_name}\nDetails: {json.dumps(section_data, indent=2)}"
            docs.append(Document(page_content=section_text, metadata={"section": section_name}))
    return docs

# To clean patient_002 data
def normalize_patient_name(patient):
    """
    Normalize the patient name field to a string "First Last".
    Handles both string and dict formats.
    """

    name = patient.get("name")
    if isinstance(name, dict):
        given = name.get("given", "")
        family = name.get("family", "")
        return f"{given} {family}".strip()
    elif isinstance(name, str):
        return name
    else:
        return "Unknown"


# Extract available patient IDs from dataset
def get_patient_ids(ehr_data):
    """
    Finds all unique patient entries and returns their IDs and names.
    """
    patients = []
    for record in ehr_data:
        patient = record.get("patient")
        if patient:
            pid = patient.get("patient_id", "UnknownID")
            name = normalize_patient_name(patient)
            patients.append((pid, name))
    return patients


# Get records only for selected patient
def filter_by_patient(full_data, patient_id):
    """
    Filters the full dataset to return all records associated with a patient_id.
    """
    patient_data = []
    is_correct_patient = False
    for record in full_data:
        if "patient" in record:
            if record["patient"]["patient_id"] == patient_id:
                is_correct_patient = True
                patient_data.append(record)
            else:
                is_correct_patient = False
        elif is_correct_patient:
            patient_data.append(record)
    return patient_data

# Load ESC guidelines PDF
# Extract text from PDF file
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    return "\n".join(page.get_text() for page in doc)


# Prepare vector store for guidelines
def prepare_guideline_vectorstore(guideline_text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.create_documents([guideline_text])
    embeddings = OpenAIEmbeddings()
    return FAISS.from_documents(docs, embeddings)


# Load and prepare the guideline vector store
@st.cache_resource
def load_guideline_vectorstore():
    text = extract_text_from_pdf("ehab368.pdf")
    return prepare_guideline_vectorstore(text)


# Combine patient and guideline vector stores into a single retriever
def combine_retrievers(patient_vectordb, guideline_vectordb):
    return MergerRetriever(retrievers=[
        patient_vectordb.as_retriever(),
        guideline_vectordb.as_retriever()
    ])


# Initialize LLM + VectorStore
@st.cache_resource
def load_vector_store(patient_data):
    flat_docs = flatten_ehr_json(patient_data)
    if not flat_docs:
        return None
    embeddings = OpenAIEmbeddings()
    vectordb = FAISS.from_documents(flat_docs, embeddings)
    return vectordb


## -- Streamlit App --

def main():
    st.title("Clinician EHR Assistant")
    st.markdown("**Ask questions about a specific patient's medical record and  ESC guidelines**")

    with open ("data.json", "r") as f:
        ehr_data = json.load(f)

    patient_options = get_patient_ids(ehr_data)
    
    if not patient_options:
        st.error("No patient records found in the data.")
        return

    # Create dropdown options like "001 - John Doe"
    # Combining Patient ID and Name to avoid duplicates
    #  
    display_options = [f"{pid} - {name}" for pid, name in patient_options]
    selected_display = st.selectbox("Select a patient:", display_options)

    # Extract the ID from the selected option
    selected_pid = selected_display.split(" - ")[0]

    with st.spinner(f"Indexing records for patient {selected_pid}..."):
        
        # Filter the combined data to get records for only the selected patient
        patient_data = filter_by_patient(ehr_data, selected_pid)
        
        if not patient_data:
            st.error(f"Could not find any data for patient {selected_pid}.")
            return

        # Create the vector store from the filtered patient data
        vectordb = load_vector_store(patient_data)

        if vectordb is None:
            st.warning("Could not create a searchable record for this patient.")
            return

        # Initialize the LangChain QA chain
        llm = ChatOpenAI(temperature=0, model_name="gpt-4o")

        # retriever = vectordb.as_retriever()
        guideline_db = load_guideline_vectorstore()
        combined_retriever = combine_retrievers(vectordb, guideline_db)
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=combined_retriever)

        # summary - patient overview
        with st.spinner("\n\nPatient summary:"):
            try:
                summary = qa_chain.run("""You are a clinical assistant. Based on the patient's medical record, provide a brief 2-line summary about patient
                                        that could help the clinician to browse through before every patient visit.
                                        recent medical history, and important key clinical observations in a concise paragraph for a clinician. 
                                        Hihglight important details in **bold** text.
                                        Include patient's age and gender. 
                                        Structure output to easily redable manner with buller points.
                                       
                                       Follow the format for patient details:

                                        **Patient Name**: John Doe
                                       
                                        **Age**: 45
                                       
                                        **Gender**: Male""")
                
                st.success("** Patient Summary: **")
                st.write(summary)
            except Exception as e:
                st.error(f"An error occurred while generating the summary: {e}")


    query = st.text_input("\n\n**Ask a question about the selected patient:**", placeholder="example: What is John Doe's blood pressure?")

    if query:
        with st.spinner("Generating answer..."):
            try:
                response = qa_chain.run(query)
                st.success("Answer:")
                st.write(response)
            except Exception as e:
                st.error(f"An error occurred: {e}")


if __name__ == "__main__":
    if "OPENAI_API_KEY" not in os.environ:
        st.error("Please set your OPENAI_API_KEY as an environment variable to run this app.")
    else:
        main()