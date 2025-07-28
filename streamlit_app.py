import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from retrieval import RetrievalPipeline
from augmentation import generate_response
from langchain_groq import ChatGroq
import os

@st.cache_resource
def load_saved_vector_db(index_path="faiss_index"):
    """Load the saved FAISS vector database directly"""
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Vector database not found at: {index_path}")
    
    # Initialize the same embedding model used during indexing
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Load the saved vector database
    vector_db = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    return vector_db

@st.cache_resource
def initialize_pipeline():
    """Initialize the RAG pipeline components"""
    try:
        # Load vector database
        vector_db = load_saved_vector_db("vector_store")
        
        # Initialize LLM
        llm = ChatGroq(model='gemma2-9b-it', api_key = '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')
        
        # Create retrieval pipeline
        pipeline = RetrievalPipeline(
            vector_db=vector_db,
            llm=llm,
            k_documents=8,
            diversity_threshold=0.6
        )
        
        return pipeline, True, "Pipeline initialized successfully"
    except FileNotFoundError:
        return None, False, "Vector database not found. Please run 'python indexing.py' first."
    except Exception as e:
        return None, False, f"Error initializing pipeline: {str(e)}"

def process_query(pipeline, user_query):
    """Process user query through the RAG pipeline"""
    try:
        # Get retrieval results
        results = pipeline.retrieve_with_all_steps(user_query)
        
        if not results:
            return "No relevant documents found for your query.", []
        
        # Combine retrieved content with source information
        context = "\n\n".join([
            f"Document {result['rank']} (Source: {result['source']}): {result['content']}"
            for result in results
        ])
        
        # Generate augmented response
        final_response = generate_response(user_query, context)
        
        return final_response, results
        
    except Exception as e:
        return f"Error processing query: {str(e)}", []

# Streamlit App
def main():
    st.set_page_config(
        page_title="RAG Document Query System",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("📚 RAG Document Query System")
    st.markdown("Ask questions about your documents and get AI-powered answers with source citations.")
    
    # Initialize pipeline
    pipeline, success, message = initialize_pipeline()
    
    if not success:
        st.error(message)
        st.info("Please ensure you have:")
        st.markdown("""
        1. Updated the `files_path` in `indexing.py`
        2. Run `python indexing.py` to create the vector database
        3. Installed all required dependencies
        """)
        return
    
    st.success(message)
    
    # Query input
    user_query = st.text_input(
        "Enter your query here:",
        placeholder="What are the environmental benefits of solar energy?",
        help="Ask any question about the documents in your knowledge base"
    )
    
    # Process query when submitted
    if st.button("Get Answer", type="primary") or user_query:
        if user_query.strip():
            with st.spinner("Processing your query..."):
                response, sources = process_query(pipeline, user_query)
            
            # Display response
            st.subheader("📝 Answer:")
            st.write(response)
            
            # Display sources if available
            if sources:
                st.subheader("📚 Sources:")
                for i, result in enumerate(sources, 1):
                    with st.expander(f"Source {i}: {result['source']}"):
                        st.write(f"**Processing:** {result['processing_steps']}")
                        st.write(f"**Content Preview:**")
                        st.write(result['content'][:500] + "..." if len(result['content']) > 500 else result['content'])
        else:
            st.warning("Please enter a query to get started.")
    
    # Sidebar with information
    with st.sidebar:
        st.markdown("### 🔧 System Information")
        st.info("""
        This RAG system uses:
        - **Multi-query retrieval** for better document discovery
        - **MMR (Maximal Marginal Relevance)** for diverse results
        - **Contextual compression** for relevant content extraction
        - **Document reordering** for optimal context utilization
        """)
        
        st.markdown("### 💡 Tips")
        st.markdown("""
        - Be specific in your questions
        - Try different phrasings if you don't get good results
        - Questions about topics covered in your documents work best
        """)

if __name__ == "__main__":
    main()