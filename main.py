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

# New imports for AI Agents
from langchain.agents import Tool, AgentExecutor, ZeroShotAgent
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

# Azure Form Recognizer endpoint and API key
endpoint = os.getenv('ENDPOINT')
key = os.getenv('KEY')

# LLM Credentials
llm_key = os.getenv('GROQ_API_KEY')

# Initialize the DocumentAnalysisClient
credential = AzureKeyCredential(key)
document_analysis_client = DocumentAnalysisClient(endpoint, credential)

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

# =================== NEW AI AGENT FUNCTIONS ===================

# Function for Composition Analysis Agent
def analyze_composition(composition_text):
    """Analyze the chemical composition of a steel alloy"""
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
    # Extract document sections for targeted analysis
    sections = extract_document_sections(full_text)
    
    # Identify standards mentioned in the document
    standards_mentioned = re.findall(r'(?:ASTM|ISO|EN|DIN|JIS|SAE|AISI|API|ASME)\s*[-:A-Z0-9]+', full_text)
    standards_list = ", ".join(standards_mentioned) if standards_mentioned else "No specific standards identified"
    
    # Run specialized analyses
    composition_analysis = analyze_composition(sections['chemical_composition'])
    properties_analysis = analyze_mechanical_properties(sections['mechanical_properties'])
    standards_analysis = analyze_standards_compliance(full_text, standards_list)
    applications_analysis = recommend_applications(full_text)
    
    # Combine all analyses into a comprehensive report
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
    return orchestration_chain.invoke({
        "composition_analysis": composition_analysis,
        "properties_analysis": properties_analysis,
        "standards_analysis": standards_analysis,
        "applications_analysis": applications_analysis
    })["text"]

# Function to handle interactive queries
def answer_query(query, document_text, memory):
    """Handle interactive queries about the steel alloy"""
    
    query_prompt = PromptTemplate.from_template("""
    You are a technical expert on steel alloys who has thoroughly analyzed the following document:
    
    {document_text}
    
    Previous conversation:
    {chat_history}
    
    User Question: {query}
    
    Answer the question based on the document content. If specific information is not available in the document,
    state this clearly but provide relevant metallurgical knowledge that might help the user.
    Keep your answer concise, accurate, and focused on the question.
    """)
    
    # FIX: Change the ConversationChain initialization 
    conversation = ConversationChain(
        llm=llm,
        prompt=query_prompt,
        memory=memory,
        verbose=True,
        # Add input variables that aren't provided by memory
        input_variables=["document_text", "query"]
    )
    
    # FIX: Change how we call the chain
    response = conversation({"document_text": document_text, "query": query})
    return response["response"]  # Extract the response from the result

# =================== STREAMLIT UI ===================
# Streamlit UI
st.title("Steel Alloy AI Analysis System")

# Sidebar for logo and file uploader
st.sidebar.image("company_logo.png", use_container_width=True)
uploaded_file = st.sidebar.file_uploader("Upload Data document", type="pdf")
uploaded_template_file = st.sidebar.file_uploader("Upload template", type='pdf')

# Initialize tabs
tab1, tab2, tab3 = st.tabs(["Document Processing", "AI Analysis", "Interactive Q&A"])

# Initialize llm
llm = ChatGroq(temperature=0, model_name="llama-3.3-70b-versatile")

# Placeholder for extracted text
extracted_text = ""
pdf_path = None

with tab1:
    st.header("Document Processing")
    
    if uploaded_file is not None:
        st.subheader("File Preview")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            pdf_path = tmp_file.name

        st.markdown(f'<iframe src="{pdf_to_base64(pdf_path)}" width="600" height="500"></iframe>', unsafe_allow_html=True)

        if st.button("Extract Content"):
            with st.spinner("Processing document..."):
                with open(pdf_path, "rb") as f:
                    analyze_result = document_analysis_client.begin_analyze_document("prebuilt-layout", document=f).result()

                extracted_text = analyze_result.content
                st.session_state['extracted_text'] = extracted_text
                modified_text = clean_text(extracted_text)
                st.session_state['modified_text'] = modified_text

                # Extract key-value pairs from the OCR content
                key_value_pairs = extract_key_value_pairs(extracted_text)
                st.session_state['key_value_pairs'] = key_value_pairs

                extracted_json_path = "extracted_key_value_pairs.json"
                with open(extracted_json_path, "w") as file:
                    json.dump(key_value_pairs, file, indent=4)

                st.write("")  # Add space
                st.subheader("Original Content")
                st.text_area("Raw Content", value=extracted_text, height=300)

                st.subheader("Extracted Key-Value Pairs")
                st.code(json.dumps(key_value_pairs, indent=4), language='json', line_numbers=False)

            # LLM to generate a basic Technical Formula
            with st.spinner("Generating Technical Formula..."):
                if uploaded_template_file is not None:
                    template_bytes = uploaded_template_file.read()
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_template_file:
                        tmp_template_file.write(template_bytes)
                        template_path = tmp_template_file.name
                    
                    pdf_loader = PyPDFLoader(template_path)
                    docs = pdf_loader.load()
                    prompt_template = """
                    Task: You are given a steel-alloy material specification document.
                    You are an expert in determining the technical formula used in production line to create a product and also consume the knowledge related to making steel alloy products.
                    You will create a document form the input text you are given to create a technical formula document that can be given to production line.
                    Regarding the technical formula, it is the chemical composition, Mechanical properties and the Metallurgical Characteristics used on the composition of steel alloys. 
                    Identify all the necessary components for the composition of steel alloys to make a technical formula professional document with all the needed names, dates, revisions, standards etc.
                    Input: {template} {text}
                    Output: A markdown formatted professional looking technical formula document. 
                    Instructions: 
                    Do not go out of the context and keep your response concise and precise. 
                    Do not make up information that is not there in the document provided. 
                    """
                    prompt = PromptTemplate.from_template(prompt_template)
                    llm_chain = LLMChain(llm=llm, prompt=prompt)
                    result = llm_chain.invoke(input={'template': docs, 'text': modified_text})
                    llm_output = result['text']
                    st.session_state['formula_output'] = llm_output
                else:
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
                    prompt = PromptTemplate.from_template(prompt_template)
                    llm_chain = LLMChain(llm=llm, prompt=prompt)
                    result = llm_chain.invoke(modified_text)
                    llm_output = result['text']
                    st.session_state['formula_output'] = llm_output
                
                st.subheader("Basic Technical Formula")
                st.markdown(llm_output)
                
                pdf_file = generate_pdf_file(llm_output)
                if pdf_file:
                    st.download_button(
                        label="Download Basic Technical Formula as PDF",
                        data=pdf_file,
                        file_name="basic_technical_formula.pdf",
                        mime="application/pdf"
                    )

with tab2:
    st.header("AI Agent Analysis")
    
    if 'extracted_text' in st.session_state and st.session_state['extracted_text']:
        if st.button("Run AI Agents Analysis"):
            with st.spinner("AI Agents analyzing document..."):
                # Create memory for the agents
                memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
                
                # Run orchestrated analysis
                orchestrated_analysis = orchestrate_analysis(
                    st.session_state['extracted_text'], 
                    st.session_state['key_value_pairs']
                )
                
                # Store results in session state
                st.session_state['orchestrated_analysis'] = orchestrated_analysis
                
                # Display comprehensive analysis
                st.subheader("Comprehensive AI Analysis")
                st.markdown(orchestrated_analysis)
                
                # Generate PDF for the comprehensive analysis
                analysis_pdf = generate_pdf_file(orchestrated_analysis)
                if analysis_pdf:
                    st.download_button(
                        label="Download Comprehensive Analysis as PDF",
                        data=analysis_pdf,
                        file_name="ai_analysis_report.pdf",
                        mime="application/pdf"
                    )
    else:
        st.info("Please upload and process a document in the 'Document Processing' tab first.")

with tab3:
    st.header("Interactive Q&A")
    
    if 'extracted_text' in st.session_state and st.session_state['extracted_text']:
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
                    # Build conversation history text from previous exchanges
                    conversation_history = ""
                    for msg in st.session_state['chat_history'][:-1]:  # Exclude the current query
                        prefix = "Human: " if msg['role'] == 'user' else "AI: "
                        conversation_history += f"{prefix}{msg['content']}\n\n"
                    
                    # Create the full prompt with document and conversation context
                    full_prompt = f"""
                    You are a technical expert on steel alloys who has thoroughly analyzed the following document:
                    
                    {st.session_state['extracted_text']}
                    
                    Previous conversation:
                    {conversation_history}
                    
                    User Question: {user_query}
                    
                    Answer the question based on the document content. If specific information is not available in the document,
                    state this clearly but provide relevant metallurgical knowledge that might help the user.
                    Keep your answer concise, accurate, and focused on the question.
                    """
                    
                    # Call the LLM directly without using ConversationChain
                    response = llm.invoke(full_prompt)
                    agent_response = response.content if hasattr(response, 'content') else str(response)
                    
                    # Display AI response
                    st.chat_message('assistant').write(agent_response)
                    st.session_state['chat_history'].append({"role": "assistant", "content": agent_response})
                
                except Exception as e:
                    error_message = f"Error processing your question: {str(e)}"
                    st.error(error_message)
                    st.chat_message('assistant').write(f"I'm sorry, I encountered an error while processing your question. Please try again or rephrase your question.")
    else:
        st.info("Please upload and process a document in the 'Document Processing' tab first.")