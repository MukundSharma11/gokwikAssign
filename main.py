from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from retrieval import RetrievalPipeline
from augmentation import generate_response
from langchain_groq import ChatGroq
import os

def load_saved_vector_db(index_path="faiss_index"):
    """Load the saved FAISS vector database directly"""
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Vector database not found at: {index_path}")
    
    # Initialize the same embedding model used during indexing
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # Load the saved vector database
    vector_db = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    return vector_db

# Initialize your components
llm = ChatGroq(model='gemma2-9b-it', api_key = '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')

# Load the saved vector database directly
try:
    vector_db = load_saved_vector_db("faiss_index")
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Please run 'python indexing.py' first to create the vector database.")
    exit(1)
except Exception as e:
    print(f"Error loading vector database: {e}")
    exit(1)

# Create the retrieval pipeline
pipeline = RetrievalPipeline(
    vector_db=vector_db,
    llm=llm,
    k_documents=8,  # Number of final documents to return
    diversity_threshold=0.6  # 0=max diversity, 1=max relevance
)

# Use the pipeline
user_query = "What are the environmental benefits of solar energy?"
results = pipeline.retrieve_with_all_steps(user_query)

if not results:
    print("No results found. Please check your index and query.")
    exit(1)

# Access the results
print("Retrieved Documents:")
for result in results:
    print(f"Rank {result['rank']}: {result['source']}")
    print(f"Content: {result['content'][:200]}...")
    print(f"Processing: {result['processing_steps']}")
    print("-" * 50)

# Step 2: Combine retrieved content with source information
context = "\n\n".join([
    f"Document {result['rank']} (Source: {result['source']}): {result['content']}"
    for result in results
])

# Step 3: Generate augmented response with source citations
final_response = generate_response(user_query, context)

# Step 4: Display results with source information
print("\nFinal RAG Response:")
print(final_response)
print("\n" + "="*50)
print("Retrieved Sources:")
for result in results:
    print(f"• {result['source']}: {result['content'][:100]}...")