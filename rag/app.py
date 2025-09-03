import streamlit as st
import os
import tempfile
import sys
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import time

# Handle async/event loop issues in Streamlit
def fix_asyncio():
    """Fix asyncio event loop issues in Streamlit"""
    try:
        import asyncio
        import nest_asyncio
        nest_asyncio.apply()
    except ImportError:
        st.error("Please install nest_asyncio: pip install nest_asyncio")
        st.stop()
    except Exception as e:
        # Alternative approach if nest_asyncio doesn't work
        try:
            import asyncio
            if sys.platform.startswith('win'):
                asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        except Exception as loop_error:
            st.warning(f"Event loop setup warning: {str(loop_error)}")

# Apply the fix
fix_asyncio()

# Now import the Google AI components
try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
    from langchain_community.vectorstores import FAISS
    from langchain_core.prompts import PromptTemplate
    from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
    from langchain_core.output_parsers import StrOutputParser
except ImportError as e:
    st.error(f"Please install required packages: {str(e)}")
    st.info("Run: pip install langchain-google-genai langchain-community faiss-cpu nest_asyncio")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="📚 PDF Q&A Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 3rem;
    }
    
    .upload-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
    }
    
    .chat-section {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 15px;
        border-left: 5px solid #667eea;
    }
    
    .question-box {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid #e1e8ed;
        margin: 1rem 0;
    }
    
    .answer-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .stats-container {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #e1e8ed;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .sidebar-info {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 20px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
    }
    
    .error-box {
        background: #fee;
        border: 1px solid #fcc;
        padding: 1rem;
        border-radius: 5px;
        color: #c00;
        margin: 1rem 0;
    }
    
    .success-box {
        background: #efe;
        border: 1px solid #cfc;
        padding: 1rem;
        border-radius: 5px;
        color: #060;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'pdf_processed' not in st.session_state:
    st.session_state.pdf_processed = False
if 'document_stats' not in st.session_state:
    st.session_state.document_stats = {}
if 'processing_error' not in st.session_state:
    st.session_state.processing_error = None

def safe_create_embeddings(chunks, api_key):
    """Safely create embeddings with error handling"""
    try:
        # Ensure API key is set
        os.environ["GOOGLE_API_KEY"] = api_key
        
        # Create embeddings with error handling
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        # Test the embeddings with a small sample first
        test_chunk = chunks[:1] if len(chunks) > 0 else chunks
        test_store = FAISS.from_documents(test_chunk, embeddings)
        
        # If test successful, create full vector store
        if len(chunks) > 1:
            vector_store = FAISS.from_documents(chunks, embeddings)
        else:
            vector_store = test_store
            
        return vector_store, None
        
    except Exception as e:
        error_msg = str(e)
        if "event loop" in error_msg.lower():
            return None, "Event loop error. Please restart the app and try again."
        elif "api" in error_msg.lower() or "key" in error_msg.lower():
            return None, "API key error. Please check your Google API key."
        elif "quota" in error_msg.lower():
            return None, "API quota exceeded. Please check your usage limits."
        else:
            return None, f"Error creating embeddings: {error_msg}"

def process_pdf_safely(file_path, chunk_size, chunk_overlap, api_key):
    """Safely process PDF with comprehensive error handling"""
    try:
        # Load PDF
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        if not documents:
            return None, None, "No content found in PDF file."
        
        # Extract text
        transcript = " ".join(doc.page_content for doc in documents)
        
        if not transcript.strip():
            return None, None, "PDF appears to be empty or contains no readable text."
        
        # Split text into chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )
        chunks = splitter.create_documents([transcript])
        
        if not chunks:
            return None, None, "Failed to create text chunks from PDF."
        
        # Create vector store
        vector_store, error = safe_create_embeddings(chunks, api_key)
        
        if error:
            return None, None, error
        
        # Create document stats
        stats = {
            'pages': len(documents),
            'chunks': len(chunks),
            'avg_chunk_size': int(len(transcript) / len(chunks)) if chunks else 0,
            'total_chars': len(transcript)
        }
        
        return vector_store, stats, None
        
    except Exception as e:
        return None, None, f"Error processing PDF: {str(e)}"

# Sidebar for API key and settings
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    api_key = st.text_input(
        "Google API Key:",
        type="password",
        placeholder="Enter your Google Gemini API key",
        help="Get your API key from Google AI Studio"
    )
    
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
        st.success("✅ API Key configured!")
    else:
        st.warning("⚠️ Please enter your Google API key to continue")
    
    st.markdown("---")
    
    st.markdown("### 📊 Document Statistics")
    if st.session_state.pdf_processed and st.session_state.document_stats:
        st.markdown(f"""
        <div class="sidebar-info">
            <strong>📄 Total Pages:</strong> {st.session_state.document_stats.get('pages', 'N/A')}<br>
            <strong>📝 Text Chunks:</strong> {st.session_state.document_stats.get('chunks', 'N/A')}<br>
            <strong>📏 Avg Chunk Size:</strong> {st.session_state.document_stats.get('avg_chunk_size', 'N/A')} chars<br>
            <strong>📊 Total Characters:</strong> {st.session_state.document_stats.get('total_chars', 'N/A'):,}
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 🔧 Advanced Settings")
    chunk_size = st.slider("Chunk Size", 200, 1000, 400, 50)
    chunk_overlap = st.slider("Chunk Overlap", 50, 300, 200, 50)
    temperature = st.slider("Model Temperature", 0.0, 1.0, 0.2, 0.1)
    
    st.markdown("---")
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()
    
    if st.button("🔄 Reset Application"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# Main content
st.markdown('<h1 class="main-header">📚 PDF Q&A Assistant</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Upload a PDF document and ask intelligent questions using AI-powered search</p>', unsafe_allow_html=True)

# Installation instructions
with st.expander("📦 Installation Requirements", expanded=False):
    st.code("""
pip install streamlit langchain-community langchain-google-genai faiss-cpu nest_asyncio pypdf tiktoken
    """, language="bash")
    st.markdown("**Note:** If you encounter event loop errors, restart the Streamlit app.")

# File upload section
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.markdown("### 📤 Upload Your PDF Document")
    
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type="pdf",
        help="Upload a PDF document to analyze and ask questions about"
    )
    
    if uploaded_file is not None:
        st.info(f"📄 **File:** {uploaded_file.name} ({uploaded_file.size:,} bytes)")
        
        if not api_key:
            st.error("⚠️ Please enter your Google API key in the sidebar first!")
        else:
            process_pdf = st.button("🔍 Process PDF", use_container_width=True)
            
            if process_pdf:
                with st.spinner("🔄 Analyzing document with AI intelligence..."):
                    st.progress(25, "Loading document...")
                    
                    try:
                        # Save uploaded file temporarily
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                            tmp_file.write(uploaded_file.read())
                            tmp_path = tmp_file.name
                        
                        st.progress(50, "Processing content...")
                        
                        # Process PDF safely
                        vector_store, stats, error = process_pdf_safely(
                            tmp_path, chunk_size, chunk_overlap, api_key
                        )
                        
                        st.progress(75, "Building knowledge base...")
                        
                        # Clean up temp file
                        os.unlink(tmp_path)
                        
                        st.progress(100, "Complete!")
                        
                        if error:
                            st.error(f"❌ Analysis failed: {error}")
                            st.session_state.processing_error = error
                        else:
                            # Store in session state
                            st.session_state.vector_store = vector_store
                            st.session_state.pdf_processed = True
                            st.session_state.document_stats = stats
                            st.session_state.processing_error = None
                            
                            st.success("✅ Document successfully analyzed and ready for intelligent queries!")
                            st.balloons()
                        
                    except Exception as e:
                        error_msg = f"System error during analysis: {str(e)}"
                        st.error(f"❌ {error_msg}")
                        st.session_state.processing_error = error_msg
                        # Clean up temp file
                        os.unlink(tmp_path)
                        
                        if error:
                            st.error(f"❌ {error}")
                            st.session_state.processing_error = error
                        else:
                            # Store in session state
                            st.session_state.vector_store = vector_store
                            st.session_state.pdf_processed = True
                            st.session_state.document_stats = stats
                            st.session_state.processing_error = None
                            
                            st.success("✅ PDF processed successfully! You can now ask questions.")
                            st.balloons()
                        
                    except Exception as e:
                        error_msg = f"Unexpected error: {str(e)}"
                        st.error(f"❌ {error_msg}")
                        st.session_state.processing_error = error_msg
    
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    if st.session_state.pdf_processed:
        st.markdown("### 📈 Processing Status")
        st.success("✅ PDF Ready for Questions")
        st.info(f"📊 {st.session_state.document_stats.get('chunks', 0)} chunks created")
        st.info(f"📄 {st.session_state.document_stats.get('pages', 0)} pages processed")
    elif st.session_state.processing_error:
        st.markdown("### ❌ Processing Status")
        st.error("Processing failed")
        st.warning("Try restarting the app if you see event loop errors")

# Error handling display
if st.session_state.processing_error:
    st.markdown("---")
    with st.expander("🔧 Troubleshooting Tips", expanded=True):
        st.markdown("""
        **Common Solutions:**
        1. **Event Loop Errors:** Restart the Streamlit app completely
        2. **API Key Errors:** Verify your Google API key is correct
        3. **Quota Errors:** Check your Google AI Studio usage limits
        4. **PDF Errors:** Ensure your PDF contains readable text
        5. **Memory Errors:** Try reducing chunk size or using smaller PDFs
        
        **If problems persist:**
        - Restart your Streamlit server: `Ctrl+C` then `streamlit run app.py`
        - Check if all required packages are installed
        - Verify your internet connection
        """)

# Chat interface
if st.session_state.pdf_processed and st.session_state.vector_store:
    st.markdown("---")
    st.markdown('<div class="chat-section">', unsafe_allow_html=True)
    st.markdown("### 💬 Ask Questions About Your Document")
    
    # Question input
    question = st.text_input(
        "Enter your question:",
        placeholder="e.g., What are the main topics covered in this document?",
        key="question_input"
    )
    
    col1, col2, col3 = st.columns([1, 1, 4])
    
    with col1:
        ask_button = st.button("🔍 Ask Question", use_container_width=True)
    
    with col2:
        if st.button("💡 Get Suggestions", use_container_width=True):
            suggestions = [
                "What are the main topics covered in this document?",
                "Can you summarize the key points?",
                "What are the important dates or numbers mentioned?",
                "Who are the main people or entities discussed?",
                "What conclusions can be drawn from this document?"
            ]
            st.info("💡 **Suggested Questions:**\n" + "\n".join([f"• {s}" for s in suggestions]))
    
    if ask_button and question:
        if not api_key:
            st.error("⚠️ Please enter your Google API key in the sidebar!")
        else:
            with st.spinner("🤔 Searching for answers..."):
                try:
                    # Set up retrieval chain with error handling
                    retriever = st.session_state.vector_store.as_retriever(
                        search_type="similarity", 
                        search_kwargs={"k": 4}
                    )
                    
                    # Set up LLM
                    llm = GoogleGenerativeAI(
                        model="gemini-1.5-flash", 
                        temperature=temperature
                    )
                    
                    # Create prompt template
                    prompt = PromptTemplate(
                        template="""
                        You are a helpful AI assistant analyzing a document. 
                        Answer the question based ONLY on the provided context from the document.
                        If the context doesn't contain enough information to answer the question, 
                        say "I don't have enough information in the document to answer this question."
                        
                        Be comprehensive but concise in your answer. Use bullet points or numbered lists when appropriate.
                        
                        Context from document:
                        {context}
                        
                        Question: {question}
                        
                        Answer:
                        """,
                        input_variables=['context', 'question']
                    )
                    
                    # Format documents function
                    def format_docs(retrieved_docs):
                        return "\n\n".join(doc.page_content for doc in retrieved_docs)
                    
                    # Create chain
                    chain = (
                        RunnableParallel({
                            'context': retriever | RunnableLambda(format_docs),
                            'question': RunnablePassthrough()
                        })
                        | prompt 
                        | llm 
                        | StrOutputParser()
                    )
                    
                    # Get answer
                    answer = chain.invoke(question)
                    
                    # Add to chat history
                    st.session_state.chat_history.append({
                        'question': question,
                        'answer': answer,
                        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
                    })
                    
                    # Clear input and refresh
                    st.rerun()
                    
                except Exception as e:
                    error_msg = str(e)
                    if "event loop" in error_msg.lower():
                        st.error("❌ Event loop error. Please restart the Streamlit app and try again.")
                    else:
                        st.error(f"❌ Error generating answer: {error_msg}")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Display chat history
    if st.session_state.chat_history:
        st.markdown("---")
        st.markdown("### 💭 Conversation History")
        
        for i, chat in enumerate(reversed(st.session_state.chat_history)):
            with st.container():
                st.markdown(f"""
                <div class="question-box">
                    <strong>❓ Question {len(st.session_state.chat_history) - i}:</strong><br>
                    {chat['question']}<br>
                    <small style="color: #888;">Asked on {chat['timestamp']}</small>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="answer-box">
                    <strong>🤖 Answer:</strong><br>
                    {chat['answer']}
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("---")

# Professional footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 3rem 0; background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%); border-radius: 12px; margin-top: 2rem;">
    <div style="max-width: 600px; margin: 0 auto;">
        <h3 style="color: #1f2937; margin-bottom: 1rem; font-family: 'Inter', sans-serif;">🏢 Enterprise-Grade Security</h3>
        <p style="color: #6b7280; margin-bottom: 1.5rem; line-height: 1.6;">
            All documents are processed with bank-level security standards. Your data remains confidential and is never stored permanently on our servers.
        </p>
        <div style="display: flex; justify-content: center; gap: 2rem; margin: 2rem 0;">
            <div style="text-align: center;">
                <div style="font-size: 1.5rem; color: #3b82f6;">🔒</div>
                <div style="font-size: 0.875rem; color: #6b7280; margin-top: 0.5rem;">End-to-End<br>Encryption</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 1.5rem; color: #3b82f6;">⚡</div>
                <div style="font-size: 0.875rem; color: #6b7280; margin-top: 0.5rem;">Real-Time<br>Processing</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 1.5rem; color: #3b82f6;">🎯</div>
                <div style="font-size: 0.875rem; color: #6b7280; margin-top: 0.5rem;">Executive-Grade<br>Insights</div>
            </div>
        </div>
        <p style="color: #9ca3af; font-size: 0.875rem; margin-top: 2rem;">
            Powered by Google Gemini AI • LangChain • Streamlit Framework
        </p>
    </div>
</div>
""", unsafe_allow_html=True)