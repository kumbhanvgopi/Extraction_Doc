from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
import os
import markdown
from xhtml2pdf import pisa
import streamlit as st
from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient
from langchain_community.document_loaders import PyPDFLoader
import pandas as pd
import re
import json
import tempfile
import base64
import io
from io import BytesIO
import time
import threading
import asyncio
import shutil
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
# from streamlit.runtime.scriptrunner.script_run_context import get_script_run_ctx
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import urllib.request
from pathlib import Path

# New imports for AI Agents
from langchain.agents import Tool, AgentExecutor, ZeroShotAgent
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

# Automation imports
import streamlit.components.v1 as components
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Azure Form Recognizer endpoint and API key
endpoint = os.getenv('ENDPOINT')
key = os.getenv('KEY')

# LLM Credentials
llm_key = os.getenv('GROQ_API_KEY')

# Initialize the DocumentAnalysisClient
credential = AzureKeyCredential(key)
document_analysis_client = DocumentAnalysisClient(endpoint, credential)

# =================== AUTOMATION CLASS ===================
class AutomationAgent:
    def __init__(self, pdf_path=None):
        self.pdf_path = pdf_path
        self.workflow_state = "init"
        self.completed_steps = []
        self.current_step = "Initialize"
        self.download_dir = os.path.join(os.getcwd(), "downloads")
        self.manual_upload = False
        
        # Create download directory if it doesn't exist
        if not os.path.exists(self.download_dir):
            os.makedirs(self.download_dir)
            
        # Set up a file watcher for the download directory
        self.observer = Observer()
        handler = DownloadHandler(self)
        self.observer.schedule(handler, self.download_dir, recursive=False)
        self.observer.start()
        
    def __del__(self):
        if hasattr(self, 'observer') and self.observer.is_alive():
            self.observer.stop()
            self.observer.join()
    
    def set_pdf_path(self, pdf_path):
        self.pdf_path = pdf_path
        self.workflow_state = "document_ready"
        self.manual_upload = True
        self.completed_steps.append("Document manually uploaded")
        
    def start_workflow(self):
        """Start the automated workflow"""
        if self.workflow_state == "init":
            self.current_step = "Getting document"
            # If no PDF is provided, download a sample PDF
            self.download_sample_pdf()
        else:
            self.execute_workflow()
    
    def download_sample_pdf(self):
        """Download a sample PDF if none is provided"""
        sample_url = "https://www.steelworld.com/pdf/0417sw28.pdf"
        local_path = os.path.join(self.download_dir, "sample_steel_alloy.pdf")
        
        try:
            urllib.request.urlretrieve(sample_url, local_path)
            self.pdf_path = local_path
            self.workflow_state = "document_ready"
            self.completed_steps.append("Document downloaded")
            self.execute_workflow()
        except Exception as e:
            st.error(f"Failed to download sample PDF: {e}")
            self.workflow_state = "error"
    
    def execute_workflow(self):
        """Execute the workflow based on current state"""
        if self.workflow_state == "document_ready":
            self.current_step = "Processing document"
            self.process_document()
        elif self.workflow_state == "document_processed":
            self.current_step = "Running AI analysis"
            self.run_ai_analysis()
        elif self.workflow_state == "analysis_complete":
            self.current_step = "Downloading report"
            self.download_report()
        elif self.workflow_state == "report_downloaded":
            self.current_step = "Ready for Q&A"
            self.setup_qa()
    
    def process_document(self):
        """Automatically process the document"""
        try:
            # Auto-upload the document by setting it in session state
            if 'uploaded_file' not in st.session_state and not self.manual_upload:
                with open(self.pdf_path, "rb") as file:
                    file_bytes = file.read()
                    # Create a file-like object
                    uploaded_file = io.BytesIO(file_bytes)
                    uploaded_file.name = os.path.basename(self.pdf_path)
                    st.session_state['uploaded_file'] = uploaded_file
            
            # Process document content automatically
            with open(self.pdf_path, "rb") as f:
                analyze_result = document_analysis_client.begin_analyze_document("prebuilt-layout", document=f).result()

            extracted_text = analyze_result.content
            st.session_state['extracted_text'] = extracted_text
            modified_text = clean_text(extracted_text)
            st.session_state['modified_text'] = modified_text
            
            # Extract key-value pairs
            key_value_pairs = extract_key_value_pairs(extracted_text)
            st.session_state['key_value_pairs'] = key_value_pairs
            
            # Save extracted JSON
            extracted_json_path = os.path.join(self.download_dir, "extracted_key_value_pairs.json")
            with open(extracted_json_path, "w") as file:
                json.dump(key_value_pairs, file, indent=4)
            
            # Generate basic technical formula
            prompt_template = """
            Task: You are given a steel-alloy material specification document.
            You are an expert in determining the technical formula used in production line to create a product and also consume the knowledge related to making steel alloy products.
            You will create a document form the input text you are given to create a technical formula document that can be given to production line.
            Regarding the technical formula, it is the chemical composition, Mechanical properties used on the composition of steel alloys. 
            Identify all the necessary components for the composition of steel alloys to make a technical formula professional document with all the needed names, dates, revisions, standards etc.
            If the values are not mentioned in the Text then just say in that place "Not mentioned". 
            Follow the below template and produce it as output.
            Input: {text}
            Output: 
            Technical Formula Template:
                    ```
                    # **Technical Formula Document - Steel Alloy [Alloy Name]**
                    - **Date**: [Date]
                    - **Review/Revision**: (if applicable)
                    - **Reference/Standard**: [Reference or Standard Number]
                    - **Product**: (if applicable)
                    - **Supplier**: [Supplier Name] (if applicable)

                    ### **Chemical Composition**:

                    ### **Mechanical Properties**:

                    ## **Additional Notes**:
                    - [Any other relevant information]

            ```
            Instructions:
            Do not go out of the context and keep your response concise and precise. 
            Only provide the information provided in the document. 
            If the chemical composition is in a tabular format, please provide your response too in a tabular format.
            If there are blanks in chemical composition, then mentioned 'not mentioned'.
            Do not make up information that is not there in the document provided. 
            Follow the template above for the structure of the document.
            Do not put any unnecessary characters.
            Keep your responses precise, concise and clear.
            """
            llm = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile")
            prompt = PromptTemplate.from_template(prompt_template)
            llm_chain = LLMChain(llm=llm, prompt=prompt)
            result = llm_chain.invoke(modified_text)
            llm_output = result['text']
            st.session_state['formula_output'] = llm_output
            
            # Generate and save PDF file
            pdf_file = generate_pdf_file(llm_output)
            if pdf_file:
                with open(os.path.join(self.download_dir, "basic_technical_formula.pdf"), "wb") as f:
                    f.write(pdf_file.getvalue())
                    
            self.workflow_state = "document_processed"
            self.completed_steps.append("Document processed")
            self.execute_workflow()
            
        except Exception as e:
            st.error(f"Error processing document: {e}")
            self.workflow_state = "error"
    
    def run_ai_analysis(self):
        """Automatically run AI analysis"""
        try:
            # Initialize active_agent in session state if not already there
            if 'active_agent' not in st.session_state:
                st.session_state['active_agent'] = None
                
            # Create memory for the agents
            memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
            
            # Run orchestrated analysis
            orchestrated_analysis = orchestrate_analysis(
                st.session_state['extracted_text'], 
                st.session_state['key_value_pairs']
            )
            
            # Store results in session state
            st.session_state['orchestrated_analysis'] = orchestrated_analysis
            
            # Generate PDF for the comprehensive analysis
            analysis_pdf = generate_pdf_file(orchestrated_analysis)
            if analysis_pdf:
                with open(os.path.join(self.download_dir, "ai_analysis_report.pdf"), "wb") as f:
                    f.write(analysis_pdf.getvalue())
            
            self.workflow_state = "analysis_complete"
            self.completed_steps.append("AI analysis completed")
            self.execute_workflow()
            
        except Exception as e:
            st.error(f"Error running AI analysis: {e}")
            self.workflow_state = "error"
    
    def download_report(self):
        """Download the generated reports"""
        # Reports should already be saved to the download directory during previous steps
        # Just confirm they exist and update state
        
        basic_formula_path = os.path.join(self.download_dir, "basic_technical_formula.pdf")
        analysis_report_path = os.path.join(self.download_dir, "ai_analysis_report.pdf")
        
        if os.path.exists(basic_formula_path) and os.path.exists(analysis_report_path):
            self.workflow_state = "report_downloaded"
            self.completed_steps.append("Reports downloaded")
            self.execute_workflow()
        else:
            st.warning("Reports not yet generated. Waiting...")
            # Schedule a retry
            threading.Timer(3.0, self.download_report).start()
    
    def setup_qa(self):
        """Set up for interactive Q&A"""
        # Initialize chat history if it doesn't exist
        if 'chat_history' not in st.session_state:
            st.session_state['chat_history'] = []
        
        # Set active tab to Q&A
        st.session_state['active_tab'] = 2  # Index for Q&A tab
        
        self.workflow_state = "qa_ready"
        self.completed_steps.append("Ready for Q&A")
        self.current_step = "Ready for queries"

# File download handler to monitor for completed downloads
class DownloadHandler(FileSystemEventHandler):
    def __init__(self, agent):
        self.agent = agent
        
    def on_created(self, event):
        if event.is_directory:
            return
        
        # If a new report file is detected, update the workflow
        if "ai_analysis_report.pdf" in event.src_path:
            self.agent.workflow_state = "report_downloaded"
            self.agent.execute_workflow()

# Function to clean and normalize the text
def clean_text(text):
    text = re.sub(r'\s*\n\s*', '\n', text)  # Remove extra spaces around new lines
    text = re.sub(r'\s*:\s*', ': ', text)   # Ensure there's a space after colons
    text = re.sub(r'\s*->\s*', ' -> ', text)  # Ensure there's a space around arrows
    text = re.sub(r'\n+', '\n', text)  # Remove multiple consecutive newlines
    return text

def extract_markdown_content(markdown_text):
    """
    Extract content between ``` markers in a string.
    
    Parameters:
    markdown_text (str): The input string containing the markers.
    
    Returns:
    str: The extracted content in Markdown format.
    """
    pattern = re.compile(r'```(.*?)```', re.DOTALL)
    match = pattern.search(markdown_text)
    
    if match:
        return match.group(1).strip()
    else:
        return markdown_text
    
def extract_key_value_pairs(text):
    # Clean the text
    text = clean_text(text)
   
    key_value_pairs = {}
    lines = text.split('\n')
    current_key = None
    current_value = []

    for line in lines:
        line = line.strip()
        # Match key-value pairs with different delimiters and formats
        if re.match(r'.*:\s.*', line) or re.match(r'.*\s*->\s*.*', line):
            if current_key:
                key_value_pairs[current_key] = ' '.join(current_value).strip()
            if ':' in line:
                parts = line.split(':', 1)
            elif '->' in line:
                parts = line.split('->', 1)
            current_key = parts[0].strip()
            current_value = [parts[1].strip()]
        else:
            if current_key:
                current_value.append(line.strip())

    if current_key:
        key_value_pairs[current_key] = ' '.join(current_value).strip()

    return key_value_pairs

# Function to convert PDF file to base64 string
def pdf_to_base64(pdf_file):
    with open(pdf_file, "rb") as f:
        pdf_bytes = f.read()
    encoded_pdf = base64.b64encode(pdf_bytes).decode("utf-8")
    return f"data:application/pdf;base64,{encoded_pdf}"

# Generate PDF from markdown content
def generate_pdf_file(llm_output):
    if '```' in llm_output:
        content = extract_markdown_content(llm_output)
    else:
        content = llm_output

    # Convert Markdown to HTML
    html_content = markdown.markdown(content, extensions=['tables'])

    # Add basic HTML structure
    full_html = f"""
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            body {{ font-family: Arial, sans-serif; }}
            table, th, td {{ border: 1px solid black; border-collapse: collapse; padding: 5px; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """

    # Convert HTML to PDF
    pdf_file = BytesIO()
    pisa_status = pisa.CreatePDF(
        src=full_html,
        dest=pdf_file
    )
    pdf_file.seek(0)

    if pisa_status.err:
        return None
    return pdf_file

# =================== AI AGENT FUNCTIONS ===================

# Function for Composition Analysis Agent
def analyze_composition(composition_text):
    """Analyze the chemical composition of a steel alloy"""
    llm = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile")
    composition_prompt = PromptTemplate.from_template("""
    You are a metallurgical expert specializing in steel alloy chemical compositions.
    
    Analyze the following steel alloy chemical composition:
    {composition_text}
    
    Provide the following analysis:
    1. Identify the key elements and their significance
    2. Highlight any unusual elements or concentrations
    3. Explain potential effects on the alloy's properties
    4. Identify the likely alloy type/classification based on composition
    5. Note any potential concerns for manufacturing or application
    
    Output your analysis in a structured, professional format.
    """)
    
    composition_chain = LLMChain(llm=llm, prompt=composition_prompt)
    return composition_chain.invoke({"composition_text": composition_text})["text"]

# Function for Mechanical Properties Agent
def analyze_mechanical_properties(properties_text):
    """Analyze the mechanical properties of a steel alloy"""
    llm = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile")
    properties_prompt = PromptTemplate.from_template("""
    You are a materials science expert specializing in steel alloy mechanical properties.
    
    Analyze the following steel alloy mechanical properties:
    {properties_text}
    
    Provide the following analysis:
    1. Evaluate the tensile strength, yield strength, and hardness values
    2. Assess ductility, toughness, and other reported properties
    3. Compare these properties to typical ranges for industrial steel alloys
    4. Identify suitable applications based on these properties
    5. Note any unusual or concerning values that require attention
    
    Output your analysis in a structured, professional format.
    """)
    
    properties_chain = LLMChain(llm=llm, prompt=properties_prompt)
    return properties_chain.invoke({"properties_text": properties_text})["text"]

# Function for Standards Compliance Agent
def analyze_standards_compliance(alloy_text, standards_mentioned):
    """Analyze the compliance with industry standards"""
    llm = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile")
    standards_prompt = PromptTemplate.from_template("""
    You are a steel industry standards expert with deep knowledge of global metallurgical standards.
    
    Analyze the following steel alloy specification for standards compliance:
    {alloy_text}
    
    Standards mentioned in the document: {standards_mentioned}
    
    Provide the following analysis:
    1. Identify all mentioned standards (ASTM, ISO, EN, etc.)
    2. Explain what each standard covers and requires
    3. Assess if the provided specifications meet those standards
    4. Note any gaps or missing information needed for full compliance
    5. Recommend additional tests or data if needed for certification
    
    Output your analysis in a structured, professional format.
    """)
    
    standards_chain = LLMChain(llm=llm, prompt=standards_prompt)
    return standards_chain.invoke({"alloy_text": alloy_text, "standards_mentioned": standards_mentioned})["text"]

# Function for Application Recommendations Agent
def recommend_applications(alloy_text):
    """Recommend suitable applications for the steel alloy"""
    llm = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile")
    applications_prompt = PromptTemplate.from_template("""
    You are an engineering expert who specializes in material selection for industrial applications.
    
    Based on the following steel alloy specification:
    {alloy_text}
    
    Provide the following recommendations:
    1. List the most suitable industrial applications for this alloy
    2. Explain why this alloy would perform well in each application
    3. Identify environmental conditions where this alloy would excel
    4. Note any applications where this alloy should NOT be used
    5. Suggest any specific treatments or processing that might enhance performance
    
    Output your recommendations in a structured, professional format.
    """)
    
    applications_chain = LLMChain(llm=llm, prompt=applications_prompt)
    return applications_chain.invoke({"alloy_text": alloy_text})["text"]

# Function to extract key sections from the document
def extract_document_sections(text):
    """Extracts key sections from the document for targeted analysis"""
    sections = {}
    
    # Extract chemical composition section
    composition_pattern = re.compile(r'(?:chemical\s+composition|composition|chemistry)(?:[^\n]*\n){1,20}?(?=\n\s*\n|\Z)', re.IGNORECASE)
    composition_match = composition_pattern.search(text)
    if composition_match:
        sections['chemical_composition'] = composition_match.group(0)
    else:
        sections['chemical_composition'] = "No clear chemical composition section found."
    
    # Extract mechanical properties section
    properties_pattern = re.compile(r'(?:mechanical\s+properties|physical\s+properties|properties)(?:[^\n]*\n){1,20}?(?=\n\s*\n|\Z)', re.IGNORECASE)
    properties_match = properties_pattern.search(text)
    if properties_match:
        sections['mechanical_properties'] = properties_match.group(0)
    else:
        sections['mechanical_properties'] = "No clear mechanical properties section found."
    
    # Extract standards section
    standards_pattern = re.compile(r'(?:standards?|specifications?|requirements|complian[tc]e)(?:[^\n]*\n){1,10}?(?=\n\s*\n|\Z)', re.IGNORECASE)
    standards_match = standards_pattern.search(text)
    if standards_match:
        sections['standards'] = standards_match.group(0)
    else:
        sections['standards'] = "No clear standards section found."
    
    return sections

# Function for Orchestrator Agent
def orchestrate_analysis(full_text, key_value_pairs):
    """Coordinate the analysis across multiple specialized agents"""
    llm = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile")
    # Extract document sections for targeted analysis
    sections = extract_document_sections(full_text)
    
    # Identify standards mentioned in the document
    standards_mentioned = re.findall(r'(?:ASTM|ISO|EN|DIN|JIS|SAE|AISI|API|ASME)\s*[-:A-Z0-9]+', full_text)
    standards_list = ", ".join(standards_mentioned) if standards_mentioned else "No specific standards identified"
    
    # Initialize active_agent in session state if not already there
    if 'active_agent' not in st.session_state:
        st.session_state['active_agent'] = None
    
    # Display agent status in session state
    st.session_state['active_agent'] = "Composition Analysis"
    # Run specialized analyses
    composition_analysis = analyze_composition(sections['chemical_composition'])
    
    st.session_state['active_agent'] = "Mechanical Properties Analysis"
    properties_analysis = analyze_mechanical_properties(sections['mechanical_properties'])
    
    st.session_state['active_agent'] = "Standards Compliance Analysis"
    standards_analysis = analyze_standards_compliance(full_text, standards_list)
    
    st.session_state['active_agent'] = "Applications Recommendations"
    applications_analysis = recommend_applications(full_text)
    
    # Combine all analyses into a comprehensive report
    st.session_state['active_agent'] = "Orchestration Agent"
    orchestration_prompt = PromptTemplate.from_template("""
    You are a senior metallurgical engineer who creates comprehensive technical reports.
    
    Integrate the following specialized analyses into a cohesive technical report:
    
    CHEMICAL COMPOSITION ANALYSIS:
    {composition_analysis}
    
    MECHANICAL PROPERTIES ANALYSIS:
    {properties_analysis}
    
    STANDARDS COMPLIANCE ANALYSIS:
    {standards_analysis}
    
    APPLICATIONS RECOMMENDATIONS:
    {applications_analysis}
    
    Create a well-structured technical report that integrates all these analyses while eliminating redundancy.
    Include an executive summary at the beginning that highlights the key findings.
    Format the output as markdown.
    """)
    
    orchestration_chain = LLMChain(llm=llm, prompt=orchestration_prompt)
    result = orchestration_chain.invoke({
        "composition_analysis": composition_analysis,
        "properties_analysis": properties_analysis,
        "standards_analysis": standards_analysis,
        "applications_analysis": applications_analysis
    })["text"]
    
    # Clear the active agent when done
    st.session_state['active_agent'] = None
    
    return result

# Function to handle interactive queries
def answer_query(query, document_text):
    """Handle interactive queries about the steel alloy"""
    llm = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile")
    
    # Build conversation history text from previous exchanges
    conversation_history = ""
    if 'chat_history' in st.session_state:
        for msg in st.session_state['chat_history']:
            prefix = "Human: " if msg['role'] == 'user' else "AI: "
            conversation_history += f"{prefix}{msg['content']}\n\n"
    
    # Create the full prompt with document and conversation context
    full_prompt = f"""
    You are a technical expert on steel alloys who has thoroughly analyzed the following document:
    
    {document_text}
    
    Previous conversation:
    {conversation_history}
    
    User Question: {query}
    
    Answer the question based on the document content. If specific information is not available in the document,
    state this clearly but provide relevant metallurgical knowledge that might help the user.
    Keep your answer concise, accurate, and focused on the question.
    """
    
    # Call the LLM directly
    response = llm.invoke(full_prompt)
    agent_response = response.content if hasattr(response, 'content') else str(response)
    
    return agent_response

# =================== STREAMLIT UI ===================
# Streamlit UI
# =================== STREAMLIT UI ===================
def main():
    st.title("Steel Alloy AI Analysis System")
    
    # Initialize session state variables if they don't exist
    if 'agent' not in st.session_state:
        st.session_state['agent'] = AutomationAgent()
    
    if 'active_tab' not in st.session_state:
        st.session_state['active_tab'] = 0
        
    if 'active_agent' not in st.session_state:
        st.session_state['active_agent'] = None
    
    # Sidebar for agent status and controls
    st.sidebar.image("company_logo.png", use_container_width=True)
    st.sidebar.title("AI Agent Status")
    st.sidebar.write(f"Current step: {st.session_state['agent'].current_step}")
    
    # Display active AI agent if one is running
    if st.session_state.get('active_agent'):
        st.sidebar.markdown(f"""
        <div style="background-color:#f0f2f6;padding:10px;border-radius:5px;margin-bottom:10px;">
            <h4 style="color:#3366ff;margin:0;">Active AI Agent:</h4>
            <p style="font-weight:bold;margin:5px 0 0 0;">{st.session_state['active_agent']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Progress bar
    completed_steps = st.session_state['agent'].completed_steps
    if completed_steps:
        st.sidebar.write("Completed steps:")
        for i, step in enumerate(completed_steps):
            st.sidebar.write(f"{i+1}. {step} ✓")
    
    # Add a start button for manual control
    if st.session_state['agent'].workflow_state == "init":
        if st.sidebar.button("Start Automated Analysis"):
            st.session_state['agent'].start_workflow()
    
    # File uploader (for manual document upload)
    uploaded_file = st.sidebar.file_uploader("Upload Data document", type="pdf")
    uploaded_template_file = st.sidebar.file_uploader("Upload template", type='pdf')
    
    # Handle manual document upload
    if uploaded_file is not None and st.session_state['agent'].workflow_state == "init":
        # Save the uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            manual_pdf_path = tmp_file.name
        
        # Update the agent with the manually uploaded file
        st.session_state['agent'].set_pdf_path(manual_pdf_path)
        st.session_state['agent'].execute_workflow()
    
    # Auto-start the agent if not already started and no manual upload
    if st.session_state['agent'].workflow_state == "init" and uploaded_file is None and st.sidebar.button("Use Sample Document"):
        st.session_state['agent'].start_workflow()
    
    # Create tabs container
    tab1, tab2, tab3 = st.tabs(["Document Processing", "AI Analysis", "Interactive Q&A"])
    
    # Override tab selection based on agent's workflow state
    if st.session_state['agent'].workflow_state == "document_processed":
        st.session_state['active_tab'] = 1
    elif st.session_state['agent'].workflow_state == "qa_ready":
        st.session_state['active_tab'] = 2
    
    # Document Processing Tab
    with tab1:
        st.header("Document Processing")
        
        # Show document preview if available
        if st.session_state['agent'].pdf_path:
            st.subheader("File Preview")
            st.markdown(f'<iframe src="{pdf_to_base64(st.session_state["agent"].pdf_path)}" width="600" height="500"></iframe>', unsafe_allow_html=True)
        
        # Show extracted content if available
        if 'extracted_text' in st.session_state:
            st.subheader("Original Content")
            st.text_area("Raw Content", value=st.session_state['extracted_text'], height=300)
            
            if 'key_value_pairs' in st.session_state:
                st.subheader("Extracted Key-Value Pairs")
                st.code(json.dumps(st.session_state['key_value_pairs'], indent=4), language='json', line_numbers=False)
            
            if 'formula_output' in st.session_state:
                st.subheader("Basic Technical Formula")
                st.markdown(st.session_state['formula_output'])
    
    # AI Analysis Tab
    with tab2:
        st.header("AI Agent Analysis")
        
        # Display active AI agent in the main panel too
        if st.session_state.get('active_agent'):
            st.info(f"🤖 Current AI Agent: **{st.session_state['active_agent']}** is analyzing the document...")
            
            # Add an animated progress indicator
            progress_html = """
            <div style="display:flex;justify-content:center;margin:20px 0;">
                <div style="border:8px solid #f3f3f3;border-top:8px solid #3498db;border-radius:50%;width:60px;height:60px;animation:spin 1s linear infinite;"></div>
            </div>
            <style>
                @keyframes spin {
                    0% { transform: rotate(0deg); }
                    100% { transform: rotate(360deg); }
                }
            </style>
            """
            st.markdown(progress_html, unsafe_allow_html=True)
        
        if st.session_state['agent'].workflow_state in ["document_processed", "analysis_complete", "report_downloaded", "qa_ready"]:
            if 'orchestrated_analysis' in st.session_state:
                st.subheader("Comprehensive AI Analysis")
                st.markdown(st.session_state['orchestrated_analysis'])
                
                # Generate download button for analysis report
                if st.session_state['agent'].workflow_state in ["report_downloaded", "qa_ready"]:
                    analysis_report_path = os.path.join(st.session_state['agent'].download_dir, "ai_analysis_report.pdf")
                    if os.path.exists(analysis_report_path):
                        with open(analysis_report_path, "rb") as f:
                            st.download_button(
                                label="Download Comprehensive Analysis as PDF",
                                data=f,
                                file_name="ai_analysis_report.pdf",
                                mime="application/pdf"
                            )
            else:
                # Manual trigger for analysis
                if st.button("Run AI Agents Analysis"):
                    st.session_state['agent'].run_ai_analysis()
                else:
                    if not st.session_state.get('active_agent'):
                        st.info("Click 'Run AI Agents Analysis' to begin the automated analysis process.")
        else:
            st.info("Document processing must be completed first.")
    
    # Interactive Q&A Tab
    with tab3:
        st.header("Interactive Q&A")
        
        if st.session_state['agent'].workflow_state in ["qa_ready"]:
            # Initialize chat history if it doesn't exist
            if 'chat_history' not in st.session_state:
                st.session_state['chat_history'] = []
                
            # Display chat history
            for message in st.session_state['chat_history']:
                if message['role'] == 'user':
                    st.chat_message('user').write(message['content'])
                else:
                    st.chat_message('assistant').write(message['content'])
            
            # Input for new questions
            user_query = st.chat_input("Ask a question about this steel alloy...")
            
            if user_query:
                # Display user message
                st.chat_message('user').write(user_query)
                st.session_state['chat_history'].append({"role": "user", "content": user_query})
                
                # Get AI response
                with st.spinner("Analyzing your question..."):
                    try:
                        agent_response = answer_query(user_query, st.session_state['extracted_text'])
                        
                        # Display AI response
                        st.chat_message('assistant').write(agent_response)
                        st.session_state['chat_history'].append({"role": "assistant", "content": agent_response})
                    
                    except Exception as e:
                        error_message = f"Error processing your question: {str(e)}"
                        st.error(error_message)
                        st.chat_message('assistant').write(f"I'm sorry, I encountered an error while processing your question. Please try again or rephrase your question.")
        else:
            st.info("AI analysis must be completed first.")

# Run the Streamlit app
if __name__ == "__main__":
    main()