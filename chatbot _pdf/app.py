import os
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI
import tempfile
import json
import random
from datetime import datetime

# --- Set your GPT API Key ---
os.environ["OPENAI_API_KEY"] = ""

# Studio Functions
def generate_audio_overview(documents, language="English"):
    """Generate audio overview of documents"""
    if not documents:
        return "No documents available for audio overview generation."
    
    # Extract text from documents
    text_content = "\n".join([doc.page_content for doc in documents[:5]])  # Limit to first 5 pages
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
    
    prompt = f"""
    Create a comprehensive audio overview of the following document content in {language}.
    The overview should be conversational and suitable for audio narration.
    
    Content:
    {text_content[:2000]}...
    
    Please provide:
    1. A brief introduction to the topic
    2. Key points and main ideas
    3. Important details and examples
    4. A conclusion summarizing the main takeaways
    
    Format the response as a natural, flowing narrative suitable for audio presentation.
    """
    
    try:
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        return f"Error generating audio overview: {str(e)}"

def generate_video_overview(documents):
    """Generate video overview script"""
    if not documents:
        return "No documents available for video overview generation."
    
    text_content = "\n".join([doc.page_content for doc in documents[:5]])
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
    
    prompt = f"""
    Create a video overview script for the following content. Include visual cues and timing suggestions.
    
    Content:
    {text_content[:2000]}...
    
    Please provide:
    1. Opening hook (0-10 seconds)
    2. Main content sections with visual suggestions
    3. Key points with timing
    4. Closing summary (last 10 seconds)
    
    Format as a video script with timestamps and visual cues.
    """
    
    try:
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        return f"Error generating video overview: {str(e)}"

def generate_mind_map(documents):
    """Generate mind map structure"""
    if not documents:
        return "No documents available for mind map generation."
    
    text_content = "\n".join([doc.page_content for doc in documents[:5]])
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
    
    prompt = f"""
    Create a mind map structure for the following content. Organize information hierarchically.
    
    Content:
    {text_content[:2000]}...
    
    Please provide a JSON structure representing the mind map with:
    - Main topic (center)
    - Primary branches (major themes)
    - Secondary branches (subtopics)
    - Tertiary branches (details)
    
    Format as JSON with nested structure.
    """
    
    try:
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        return f"Error generating mind map: {str(e)}"

def generate_report(documents):
    """Generate comprehensive report"""
    if not documents:
        return "No documents available for report generation."
    
    text_content = "\n".join([doc.page_content for doc in documents])
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
    
    prompt = f"""
    Create a comprehensive report based on the following content.
    
    Content:
    {text_content[:3000]}...
    
    Please provide:
    1. Executive Summary
    2. Key Findings
    3. Detailed Analysis
    4. Conclusions
    5. Recommendations
    
    Format as a professional report with clear sections and bullet points.
    """
    
    try:
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        return f"Error generating report: {str(e)}"

def generate_flashcards(documents):
    """Generate flashcards"""
    if not documents:
        return "No documents available for flashcard generation."
    
    text_content = "\n".join([doc.page_content for doc in documents[:5]])
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
    
    prompt = f"""
    Create flashcards based on the following content. Generate 10-15 flashcards.
    
    Content:
    {text_content[:2000]}...
    
    Format each flashcard as:
    Front: [Question or term]
    Back: [Answer or definition]
    
    Provide flashcards covering the most important concepts.
    """
    
    try:
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        return f"Error generating flashcards: {str(e)}"

def generate_quiz(documents):
    """Generate quiz questions"""
    if not documents:
        return "No documents available for quiz generation."
    
    text_content = "\n".join([doc.page_content for doc in documents[:5]])
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0.3)
    
    prompt = f"""
    Create a quiz based on the following content. Generate 10 multiple choice questions.
    
    Content:
    {text_content[:2000]}...
    
    Format each question as:
    Question: [Question text]
    A) [Option 1]
    B) [Option 2]
    C) [Option 3]
    D) [Option 4]
    Answer: [Correct answer]
    Explanation: [Brief explanation]
    
    Include questions of varying difficulty levels.
    """
    
    try:
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        return f"Error generating quiz: {str(e)}"

# Custom CSS for the enhanced professional UI design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1rem 2rem;
        background: white;
        border-bottom: 1px solid #e5e7eb;
        margin-bottom: 0;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    
    .logo-section {
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    
    .logo {
        width: 36px;
        height: 36px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: 700;
        font-size: 20px;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);
    }
    
    .header-actions {
        display: flex;
        gap: 0.5rem;
        align-items: center;
    }
    
    .action-icon {
        width: 40px;
        height: 40px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        transition: all 0.2s ease;
        color: #6b7280;
        font-size: 18px;
    }
    
    .action-icon:hover {
        background-color: #f3f4f6;
        color: #374151;
        transform: translateY(-1px);
    }
    
    .api-toggle {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        margin-right: 1rem;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    
    .api-toggle:hover {
        background: #f1f5f9;
        border-color: #cbd5e1;
    }
    
    .toggle-switch {
        position: relative;
        width: 44px;
        height: 24px;
        background: #cbd5e1;
        border-radius: 12px;
        transition: background-color 0.2s ease;
    }
    
    .toggle-switch.active {
        background: #10b981;
    }
    
    .toggle-knob {
        position: absolute;
        top: 2px;
        left: 2px;
        width: 20px;
        height: 20px;
        background: white;
        border-radius: 50%;
        transition: transform 0.2s ease;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .toggle-switch.active .toggle-knob {
        transform: translateX(20px);
    }
    
    .toggle-label {
        font-size: 14px;
        font-weight: 500;
        color: #374151;
    }
    
    .upload-header {
        padding: 1rem 2rem;
        background: #f8fafc;
        border-bottom: 1px solid #e5e7eb;
        margin-bottom: 0;
    }
    
    .upload-section {
        display: flex;
        align-items: center;
        gap: 1rem;
        max-width: 1200px;
        margin: 0 auto;
    }
    
    .upload-label {
        font-size: 16px;
        font-weight: 600;
        color: #374151;
        white-space: nowrap;
    }
    
    .two-panel-layout {
        display: flex;
        height: calc(100vh - 160px);
        gap: 0;
    }
    
    .chat-panel {
        flex: 0 0 70%;
        border-right: 1px solid #e5e7eb;
        padding: 1.5rem;
        background: white;
    }
    
    .studio-panel {
        flex: 0 0 30%;
        padding: 1.5rem;
        background: white;
    }
    
    .panel-header {
        font-size: 18px;
        font-weight: 600;
        margin-bottom: 1.5rem;
        color: #1f2937;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .add-buttons {
        display: flex;
        gap: 0.5rem;
        margin-bottom: 1.5rem;
    }
    
    .add-btn {
        padding: 0.5rem 1rem;
        border: 1px solid #d1d5db;
        border-radius: 8px;
        background: white;
        cursor: pointer;
        font-size: 14px;
        font-weight: 500;
        transition: all 0.2s ease;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .add-btn:hover {
        background: #f9fafb;
        border-color: #9ca3af;
        transform: translateY(-1px);
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .discover-btn {
        padding: 0.5rem 1rem;
        border: 1px solid #d1d5db;
        border-radius: 8px;
        background: white;
        cursor: pointer;
        font-size: 14px;
        font-weight: 500;
        transition: all 0.2s ease;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .discover-btn:hover {
        background: #f9fafb;
        border-color: #9ca3af;
        transform: translateY(-1px);
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .empty-state {
        text-align: center;
        padding: 3rem 1rem;
        color: #6b7280;
    }
    
    .document-icon {
        width: 64px;
        height: 64px;
        margin: 0 auto 1rem;
        opacity: 0.4;
        background: #f3f4f6;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 24px;
    }
    
    .upload-area {
        text-align: center;
        padding: 3rem 1rem;
        border: 2px dashed #d1d5db;
        border-radius: 12px;
        margin: 2rem 0;
        background: #fafafa;
        cursor: pointer;
        transition: all 0.2s ease;
    }
    
    .upload-area:hover {
        border-color: #9ca3af;
        background: #f9fafb;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    
    .upload-icon {
        width: 64px;
        height: 64px;
        margin: 0 auto 1rem;
        background: #f3f4f6;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 24px;
        color: #9ca3af;
    }
    
    .upload-btn {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        cursor: pointer;
        font-size: 14px;
        font-weight: 600;
        transition: all 0.2s ease;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);
    }
    
    .upload-btn:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* Chat input container removed - no longer needed */
    
    .studio-feature {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 0.75rem 1rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
        font-size: 14px;
        text-align: center;
        font-weight: 500;
        box-shadow: 0 2px 8px rgba(16, 185, 129, 0.3);
    }
    
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 0.75rem;
        margin-bottom: 1.5rem;
    }
    
    .feature-card {
        background: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        cursor: pointer;
        transition: all 0.2s ease;
        position: relative;
    }
    
    .feature-card:hover {
        background: #f3f4f6;
        border-color: #d1d5db;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    
    .feature-icon {
        width: 40px;
        height: 40px;
        margin: 0 auto 0.5rem;
        background: #f3f4f6;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 20px;
        color: #6b7280;
    }
    
    .feature-card:hover .feature-icon {
        background: #e5e7eb;
        color: #374151;
    }
    
    .feature-edit {
        position: absolute;
        top: 0.5rem;
        right: 0.5rem;
        width: 20px;
        height: 20px;
        background: #f3f4f6;
        border-radius: 4px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 12px;
        color: #9ca3af;
        opacity: 0;
        transition: opacity 0.2s ease;
    }
    
    .feature-card:hover .feature-edit {
        opacity: 1;
    }
    
    .add-note-btn {
        background: linear-gradient(135deg, #1f2937 0%, #374151 100%);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 10px;
        cursor: pointer;
        font-size: 14px;
        font-weight: 600;
        width: 100%;
        transition: all 0.2s ease;
        box-shadow: 0 2px 8px rgba(31, 41, 55, 0.3);
    }
    
    .add-note-btn:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(31, 41, 55, 0.4);
    }
    
    .studio-description {
        color: #6b7280;
        font-size: 14px;
        margin-bottom: 1rem;
        text-align: center;
        line-height: 1.5;
    }
    
    .professional-icon {
        width: 20px;
        height: 20px;
        display: inline-block;
        vertical-align: middle;
    }
    
    .studio-output {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1.5rem;
        margin-top: 1rem;
        max-height: 400px;
        overflow-y: auto;
    }
    
    .studio-output-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #e2e8f0;
    }
    
    .studio-output-title {
        font-size: 16px;
        font-weight: 600;
        color: #1f2937;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .studio-output-content {
        color: #374151;
        line-height: 1.6;
        white-space: pre-wrap;
    }
    
    .language-selector {
        margin-bottom: 1rem;
    }
    
    .loading-spinner {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid #f3f3f3;
        border-top: 3px solid #3498db;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
</style>
""", unsafe_allow_html=True)

# --- Streamlit UI ---
st.set_page_config(page_title="StudyMate", page_icon="📘", layout="wide", initial_sidebar_state="collapsed")

# Initialize session state for API toggle
if 'use_local_model' not in st.session_state:
    st.session_state.use_local_model = False

# Add a hidden toggle for API mode
# use_local = st.toggle("Local Model", value=st.session_state.use_local_model, key="api_toggle", label_visibility="collapsed")
# st.session_state.use_local_model = use_local

# Main header with professional icons and API toggle
st.markdown("""
<div class="main-header">
    <div class="logo-section">
        <div class="logo">S</div>
        <span style="font-size: 18px; font-weight: 600;">StudyMate</span>
    </div>
    <div class="header-actions">
        <div class="api-toggle" onclick="toggleApiMode()">
            <span class="toggle-label">Local Model</span>
            <div class="toggle-switch" id="apiToggle">
                <div class="toggle-knob"></div>
            </div>
        </div>
        <div class="action-icon" title="Share">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M4 12v8a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-8"></path>
                <polyline points="16,6 12,2 8,6"></polyline>
                <line x1="12" y1="2" x2="12" y2="15"></line>
            </svg>
        </div>
        <div class="action-icon" title="Settings">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <circle cx="12" cy="12" r="3"></circle>
                <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1 1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path>
            </svg>
        </div>
        <div class="action-icon" title="Apps">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <rect x="3" y="3" width="7" height="7"></rect>
                <rect x="14" y="3" width="7" height="7"></rect>
                <rect x="14" y="14" width="7" height="7"></rect>
                <rect x="3" y="14" width="7" height="7"></rect>
            </svg>
        </div>
        <div class="action-icon" title="Profile">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path>
                <circle cx="12" cy="7" r="4"></circle>
            </svg>
        </div>
    </div>
</div>

<script>
function toggleApiMode() {
    const toggle = document.getElementById('apiToggle');
    const isActive = toggle.classList.contains('active');
    
    if (isActive) {
        toggle.classList.remove('active');
        st.session_state.use_local_model = false;
    } else {
        toggle.classList.add('active');
        st.session_state.use_local_model = true;
    }
    
    // Trigger a rerun to update the session state
    window.parent.postMessage({type: 'streamlit:rerun'}, '*');
}

// Initialize toggle state
document.addEventListener('DOMContentLoaded', function() {
    const toggle = document.getElementById('apiToggle');
    if (""" + str(st.session_state.use_local_model).lower() + """) {
        toggle.classList.add('active');
    }
});
</script>
""", unsafe_allow_html=True)

# Upload section in header
st.markdown("""
<div class="upload-header">
    <div class="upload-section">
        <div class="upload-label">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="margin-right: 0.5rem;">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                <polyline points="7,10 12,15 17,10"></polyline>
                <line x1="12" y1="15" x2="12" y2="3"></line>
            </svg>
            Upload Documents
        </div>
        <div style="flex: 1;">
""", unsafe_allow_html=True)

# Upload area in header
uploaded_files = st.file_uploader(
    "Choose files to upload", 
    type=["pdf", "txt", "docx", "mp4", "mp3", "wav"], 
    accept_multiple_files=True,
    help="Upload PDFs, websites, text, videos or audio files",
    label_visibility="collapsed"
)

st.markdown("""
        </div>
        <div style="font-size: 14px; color: #6b7280;">
""", unsafe_allow_html=True)

# Show uploaded files count
if uploaded_files:
    st.markdown(f"📄 {len(uploaded_files)} file(s) uploaded")
else:
    st.markdown("No files uploaded")

st.markdown("""
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Two panel layout: Chat (70%) and Studio (30%)
col_chat, col_studio = st.columns([7, 3])

with col_chat:
    st.markdown('<div class="chat-panel">', unsafe_allow_html=True)
    st.markdown("""
    <div class="panel-header">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
        </svg>
        Chat
    </div>
    """, unsafe_allow_html=True)
    
    # Chat area
    if uploaded_files:
        st.markdown("### Chat with your documents")
        
        # Initialize chat history
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        
        # Display chat history
        for user_msg, bot_msg in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(user_msg)
            with st.chat_message("assistant"):
                st.write(bot_msg)
        
        # Chat input
        user_query = st.chat_input("Ask something about your documents...")
        if user_query:
            # Process the query (this will be handled by the PDF processing logic below)
            st.session_state.chat_history.append((user_query, "Processing your question..."))
            st.rerun()
    else:
        st.markdown("""
        <div class="empty-state">
            <div class="document-icon">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
                </svg>
            </div>
            <p>Upload files to start chatting</p>
            <p style="font-size: 12px; margin-top: 0.5rem;">Upload documents in the header to begin asking questions about your content.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

with col_studio:
    st.markdown('<div class="studio-panel">', unsafe_allow_html=True)
    st.markdown("""
    <div class="panel-header">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <rect x="2" y="3" width="20" height="14" rx="2" ry="2"></rect>
            <line x1="8" y1="21" x2="16" y2="21"></line>
            <line x1="12" y1="17" x2="12" y2="21"></line>
        </svg>
        Studio
    </div>
    """, unsafe_allow_html=True)
    
    # Language selector for Audio Overview
    languages = {
        "English": "English",
        "हिन्दी": "Hindi", 
        "বাংলা": "Bengali",
        "ગુજરાતી": "Gujarati",
        "ಕನ್ನಡ": "Kannada",
        "മലയാളം": "Malayalam",
        "मराठी": "Marathi",
        "ਪੰਜਾਬੀ": "Punjabi",
        "தமிழ்": "Tamil",
        "తెలుగు": "Telugu"
    }
    
    selected_language = st.selectbox(
        "Select language for Audio Overview:",
        options=list(languages.keys()),
        index=0,
        key="audio_language"
    )
    
    st.markdown("""
    <div class="studio-feature">
        Create an Audio Overview in: हिन्दी, বাংলা, ગુજરાતી, ಕನ್ನಡ, മലയാളം, मराठी, ਪੰਜਾਬੀ, தமிழ், తెలుగు
    </div>
    """, unsafe_allow_html=True)
    
    
    # Studio Features Grid - 2x3 layout (2 columns, 3 rows)
    col_feat1, col_feat2 = st.columns(2)
    
    with col_feat1:
        # Row 1 - Audio Overview
        if st.button("🎵 Audio Overview", key="audio_btn", help="Generate audio overview", use_container_width=True):
            if uploaded_files and "qa_chain" in st.session_state:
                with st.spinner("Generating Audio Overview..."):
                    documents = st.session_state.get("processed_docs", [])
                    result = generate_audio_overview(documents, languages[selected_language])
                    st.session_state.studio_output = {
                        "type": "Audio Overview",
                        "content": result,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "audio_text": result
                    }
            else:
                st.warning("Please upload files first!")
        
        # Row 2 - Mind Map
        if st.button("🗺️ Mind Map", key="mindmap_btn", help="Generate mind map", use_container_width=True):
            if uploaded_files and "qa_chain" in st.session_state:
                with st.spinner("Generating Mind Map..."):
                    documents = st.session_state.get("processed_docs", [])
                    result = generate_mind_map(documents)
                    st.session_state.studio_output = {
                        "type": "Mind Map",
                        "content": result,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
            else:
                st.warning("Please upload files first!")
        
        # Row 3 - Flashcards
        if st.button("🃏 Flashcards", key="flashcards_btn", help="Generate flashcards", use_container_width=True):
            if uploaded_files and "qa_chain" in st.session_state:
                with st.spinner("Generating Flashcards..."):
                    documents = st.session_state.get("processed_docs", [])
                    result = generate_flashcards(documents)
                    st.session_state.studio_output = {
                        "type": "Flashcards",
                        "content": result,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
            else:
                st.warning("Please upload files first!")
    
    with col_feat2:
        # Row 1 - Video Overview
        if st.button("🎥 Video Overview", key="video_btn", help="Generate video overview", use_container_width=True):
            if uploaded_files and "qa_chain" in st.session_state:
                with st.spinner("Generating Video Overview..."):
                    documents = st.session_state.get("processed_docs", [])
                    result = generate_video_overview(documents)
                    st.session_state.studio_output = {
                        "type": "Video Overview",
                        "content": result,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
            else:
                st.warning("Please upload files first!")
        
        # Row 2 - Reports
        if st.button("📊 Reports", key="reports_btn", help="Generate report", use_container_width=True):
            if uploaded_files and "qa_chain" in st.session_state:
                with st.spinner("Generating Report..."):
                    documents = st.session_state.get("processed_docs", [])
                    result = generate_report(documents)
                    st.session_state.studio_output = {
                        "type": "Report",
                        "content": result,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
            else:
                st.warning("Please upload files first!")
        
        # Row 3 - Quiz
        if st.button("❓ Quiz", key="quiz_btn", help="Generate quiz", use_container_width=True):
            if uploaded_files and "qa_chain" in st.session_state:
                with st.spinner("Generating Quiz..."):
                    documents = st.session_state.get("processed_docs", [])
                    result = generate_quiz(documents)
                    st.session_state.studio_output = {
                        "type": "Quiz",
                        "content": result,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
            else:
                st.warning("Please upload files first!")
    
    # Studio Output Display
    if "studio_output" in st.session_state:
        output = st.session_state.studio_output
        
        # Add audio functionality for Audio Overview
        if output['type'] == 'Audio Overview' and 'audio_text' in output:
            # Generate audio using text-to-speech
            try:
                import pyttsx3
                engine = pyttsx3.init()
                engine.setProperty('rate', 150)  # Speed of speech
                engine.setProperty('volume', 0.9)  # Volume level
                
                # Save audio to file
                audio_file = "audio_overview.wav"
                engine.save_to_file(output['audio_text'], audio_file)
                engine.runAndWait()
                
                # Display audio player
                st.audio(audio_file, format="audio/wav")
            except ImportError:
                st.info("Audio playback requires pyttsx3. Install with: pip install pyttsx3")
            except Exception as e:
                st.error(f"Audio generation failed: {str(e)}")
        
        st.markdown(f"""
        <div class="studio-output">
            <div class="studio-output-header">
                <div class="studio-output-title">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <rect x="2" y="3" width="20" height="14" rx="2" ry="2"></rect>
                        <line x1="8" y1="21" x2="16" y2="21"></line>
                        <line x1="12" y1="17" x2="12" y2="21"></line>
                    </svg>
                    {output['type']}
                </div>
                <div style="font-size: 12px; color: #6b7280;">{output['timestamp']}</div>
            </div>
            <div class="studio-output-content">{output['content']}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="studio-description">
            Studio output will be saved here. After adding sources, click to add Audio Overview, study guide, mind map and more!
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <button class="add-note-btn">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="margin-right: 0.5rem;">
            <line x1="12" y1="5" x2="12" y2="19"></line>
            <line x1="5" y1="12" x2="19" y2="12"></line>
        </svg>
        Add note
    </button>
    """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Remove the chat input at bottom - no longer needed

# PDF Processing Logic (hidden from UI)
if uploaded_files:
    # Check if we need to process new files
    if "uploaded_files" not in st.session_state or st.session_state.uploaded_files != uploaded_files:
        all_docs = []
        for uploaded_file in uploaded_files:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name

            loader = PyPDFLoader(tmp_path)
            documents = loader.load()
            all_docs.extend(documents)

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = splitter.split_documents(all_docs)

        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(docs, embeddings)

        qa = ConversationalRetrievalChain.from_llm(
            ChatOpenAI(model="gpt-4o", temperature=0),
            vectorstore.as_retriever()
        )

        # Store the QA chain and documents in session state
        st.session_state.qa_chain = qa
        st.session_state.uploaded_files = uploaded_files
        st.session_state.processed_docs = docs  # Store processed documents for studio features

# Process chat queries
if "qa_chain" in st.session_state and "chat_history" in st.session_state:
    # Check if there's a new query to process
    if st.session_state.chat_history and st.session_state.chat_history[-1][1] == "Processing your question...":
        user_query = st.session_state.chat_history[-1][0]
        try:
            result = st.session_state.qa_chain({"question": user_query, "chat_history": st.session_state.chat_history[:-1]})
            st.session_state.chat_history[-1] = (user_query, result["answer"])
        except Exception as e:
            st.session_state.chat_history[-1] = (user_query, f"Error processing query: {str(e)}")
        st.rerun()
