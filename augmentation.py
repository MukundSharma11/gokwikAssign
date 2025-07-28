from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

llm = ChatGroq(model = 'gemma2-9b-it', api_key='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')

# Enhanced augmentation prompt with source citation requirements
augmentation_prompt = PromptTemplate(
    input_variables=["query", "context"],
    template="""Based on the retrieved documents, provide a comprehensive answer to the user's question.

User Question: {query}

Retrieved Context:
{context}

Instructions:
- Use only information from the provided context
- Always cite sources using format: [Source: filename - "brief excerpt"]
- If context is insufficient, Answer: "Relevant Information to the user query missing"
- Provide a clear, well-structured response
- You may summarize or quote briefly from the provided documents
- Include a "Sources" section at the end listing all referenced files

Answer:"""
)

augmentation_chain = augmentation_prompt | llm

def generate_response(query: str, context: str) -> str:
    """Generate response using the augmentation chain"""
    return augmentation_chain.invoke({"query": query, "context": context}).content