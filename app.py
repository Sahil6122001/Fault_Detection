import streamlit as st
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# Set Gemini API key
genai.configure(api_key="AIzaSyADG5FWkseqATlmFvTRN2b7A7-EEbinLpA") 

# Streamlit setup
st.set_page_config(page_title="Telecom Fault Detection", layout="centered")
st.title("AI-Driven Telecom Fault Detection & Resolution")

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# Define telecom faults and resolutions
faults = [
    "Network latency in Zone 4",
    "Call drop in urban area",
    "4G tower not responding",
    "Slow internet speed during peak hours",
    "VoIP service interruption in Sector 12"
]

resolutions = [
    "Check routing tables and resolve latency issues.",
    "Reset call handling module in urban cluster.",
    "Ping and remotely restart the 4G base station.",
    "Inspect fiber link integrity and analyze user congestion.",
    "Restart VoIP proxy servers and verify codec compatibility."
]

# FAISS setup
fault_embeddings = model.encode(faults)
dimension = fault_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(fault_embeddings))

# Define agents
class FaultDetectorAgent:
    def run(self, query):
        query_embedding = model.encode([query])
        _, indices = index.search(np.array(query_embedding), 1)
        return faults[indices[0][0]]

class FaultAnalyzerAgent:
    def run(self, fault_description):
        prompt = f"Provide a detailed technical analysis for this telecom fault: {fault_description}"
        model = genai.GenerativeModel("models/gemini-1.5-pro")
        response = model.generate_content(prompt)
        return response.text.strip()

class ResolutionAgent:
    def run(self, fault_description):
        idx = faults.index(fault_description)
        return resolutions[idx]

class SOPAgent:
    def run(self, query):
        prompt = f"""
        You are a telecom network expert.

        Task:
        1️⃣ Identify and explain the telecom fault described: {query}
        2️⃣ Provide the SOP report including:
          - Fault description
          - Impact analysis
          - Root cause (if identifiable)
          - Resolution steps
          - Preventive measures for the future

        Make sure the report is structured, clear, and written in professional language.
        """
        gemini_model = genai.GenerativeModel(model_name="gemini-1.5-pro")
        response = gemini_model.generate_content(prompt)
        return response.text.strip()

# Telecom keyword checker
telecom_keywords = [
    "network", "signal", "call", "data", "internet", "jio", "airtel", "vi", "telecom", "coverage",
    "sms", "roaming", "4g", "5g", "mobile", "latency", "bandwidth", "tower", "connection", "sim",
    "NAT", "gaming", "ports", "router", "firewall", "network address translation", "online gaming"
]

def is_telecom_related(query):
    query = query.lower()
    return any(keyword in query for keyword in telecom_keywords)

# Streamlit user interface
st.markdown("Enter a telecom issue or question below:")
query = st.text_input("Telecom Fault or Question")

if st.button("Analyze"):
    if query.strip() == "":
        st.error("Please enter a query to proceed.")
    elif not is_telecom_related(query):
        st.error("❌ The question you asked doesn't belong to the telecom field.")
    else:
        # Run agents
        detector = FaultDetectorAgent()
        analyzer = FaultAnalyzerAgent()
        resolver = ResolutionAgent()
        sop_agent = SOPAgent()

        similar_fault = detector.run(query)
        analysis = analyzer.run(similar_fault)
        resolution = resolver.run(similar_fault)
        sop_report = sop_agent.run(query)

        # Display results
        st.subheader("Detected Similar Fault:")
        st.info(similar_fault)

        st.subheader("Technical Analysis:")
        st.write(analysis)

        st.subheader("Suggested Resolution:")
        st.success(resolution)

        st.subheader("AI-Based SOP Report:")
        st.write(sop_report)

        # Save explanation to text file
        explanation_filename = "SOP_Report.txt"
        with open(explanation_filename, "w", encoding="utf-8") as file:
            file.write(sop_report)

        # Download button
        st.download_button(
            label="Download SOP Report",
            data=open(explanation_filename, "rb").read(),
            file_name=explanation_filename,
            mime="text/plain"
        )
