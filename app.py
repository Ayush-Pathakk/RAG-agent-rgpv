import streamlit as st
import sys
from pathlib import Path
import os


# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from rag_pipeline import RAGPipeline

# Page config
st.set_page_config(
    page_title="RGPV RAG Assistant",
    page_icon="📚",
    layout="wide"
)

# Initialize RAG pipeline (cached)
@st.cache_resource
def load_pipeline():
    try:
        return RAGPipeline()
    except Exception as e:
        st.error(f"Failed to initialize pipeline: {e}")
        return None

# Title
st.title("📚 RGPV Study Assistant")
st.markdown("Ask questions from your study material - Get exam-ready answers")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    top_k = st.slider("Number of sources", 1, 5, 3)
    threshold = st.slider("Relevance threshold", 0.5, 2.0, 1.5, 0.1)
    
    st.markdown("---")
    st.markdown("**How to use:**")
    st.markdown("1. Type your question")
    st.markdown("2. Click 'Get Answer'")
    st.markdown("3. View answer + sources")
    
    st.markdown("---")
    st.info("💡 Try asking:\n- 'Previous year questions of data structure'\n- 'What is recursion?'")

# Main area
pipeline = load_pipeline()

if pipeline is None:
    st.error("❌ Failed to load RAG pipeline. Check console for errors.")
    st.stop()

# Question input
query = st.text_input("🔍 Enter your question:", placeholder="e.g., What are previous year questions of data structure?")

if st.button("Get Answer", type="primary"):
    if query.strip():
        with st.spinner("🔍 Searching knowledge base..."):
            try:
                result = pipeline.answer_question(query, top_k=top_k, score_threshold=threshold)
                
                # Display results
                if result['found']:
                    st.success("✅ Answer found!")
                    
                    # Show answer
                    st.markdown("### 📝 Answer:")
                    st.markdown(result['answer'])
                    
                    st.markdown("---")
                    
                    # Show sources
                    if result['sources']:
                        st.markdown("### 📚 Sources:")
                        for i, source in enumerate(result['sources'], 1):
                            with st.expander(f"📄 Source {i} - Page {source['page']} (Score: {source['score']:.2f})"):
                                st.write(source['text'])
                else:
                    st.warning("⚠️ No relevant information found.")
                    st.info("💡 Try:\n- Asking about 'previous year questions of data structure'\n- Rephrasing your question\n- Lowering the relevance threshold")
            
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.info("Check that:\n- Groq API key is set in .env\n- Vector store is built")
    else:
        st.error("❌ Please enter a question!")

# Footer
st.markdown("---")
st.caption("📚 RGPV RAG Study Assistant | Built with Streamlit + Groq LLM")