import streamlit as st
import os
import logging
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
import ollama

# Configure logging
logging.basicConfig(level=logging.INFO)

# Constants
DOC_PATH = ".\\data\\proposal_Tim_Berners-Lee.pdf"
MODEL_NAME = "llama3.2"
EMBEDDING_MODEL = "nomic-embed-text"
VECTOR_STORE_NAME = "simple-rag"
PERSIST_DIRECTORY = "./chroma_db"

# Load PDF documents
def ingest_pdf(doc_path):
    """Load PDF documents."""
    if os.path.exists(doc_path):
        loader = UnstructuredPDFLoader(file_path=doc_path)
        data = loader.load()
        logging.info("PDF loaded successfully.")
        return data
    else:
        logging.error(f"PDF file not found at path: {doc_path}")
        st.error("PDF file not found.")
        return None

# Split documents into smaller chunks
def split_documents(documents):
    """Split documents into smaller chunks.""" 
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
    chunks = text_splitter.split_documents(documents)
    logging.info("Documents split into chunks.")
    return chunks

@st.cache_resource
def load_vector_db():
    """Load or create the vector database with improved error handling.""" 
    try:
        ollama.pull(EMBEDDING_MODEL)
        embedding = OllamaEmbeddings(model=EMBEDDING_MODEL)

        if os.path.exists(PERSIST_DIRECTORY):
            vector_db = Chroma(
                embedding_function=embedding,
                collection_name=VECTOR_STORE_NAME,
                persist_directory=PERSIST_DIRECTORY,
            )
            logging.info("Loaded existing vector database.")
        else:
            data = ingest_pdf(DOC_PATH)
            if data is None:
                raise FileNotFoundError(f"PDF file {DOC_PATH} not found.")
                
            chunks = split_documents(data)

            vector_db = Chroma.from_documents(
                documents=chunks,
                embedding=embedding,
                collection_name=VECTOR_STORE_NAME,
                persist_directory=PERSIST_DIRECTORY,
            )
            logging.info("Vector database created and persisted.")
        return vector_db
    except FileNotFoundError as e:
        logging.error(str(e))
        st.error(f"File error: {str(e)}")
    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        st.error(f"An unexpected error occurred: {str(e)}")
    return None

# Suggest similar questions with context-awareness
def suggest_similar_questions(user_input, vector_db, llm):
    """Generate similar questions dynamically based on the user input.""" 
    suggestions = [
        f"Can you explain more about {user_input}?",
        f"What insights does the document provide about {user_input}?",
        f"Could you summarize the key points on {user_input}?",
        f"What does the document say regarding {user_input}?",
        f"How is {user_input} discussed in the document?"
    ]
    return suggestions

# Create a multi-query retriever
def create_retriever(vector_db, llm):
    """Create a multi-query retriever.""" 
    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. Your task is to generate five
different versions of the given user question to retrieve relevant documents from
a vector database. By generating multiple perspectives on the user question, your
goal is to help the user overcome some of the limitations of the distance-based
similarity search. Provide these alternative questions separated by newlines.
Original question: {question}""",
    )

    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(), llm, prompt=QUERY_PROMPT
    )
    logging.info("Retriever created.")
    return retriever

# Create the chain for answering questions
def create_chain(retriever, llm):
    """Create the chain with preserved syntax.""" 
    template = """Answer the question based ONLY on the following context:
{context}
Question: {question}
"""

    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough() }
        | prompt
        | llm
        | StrOutputParser()
    )

    logging.info("Chain created with preserved syntax.")
    return chain

# Capture feedback on the response
def capture_feedback():
    """Allow users to rate the assistant's response.""" 
    feedback = st.radio(
        "Was the answer helpful?", ("Yes", "No", "Maybe")
    )
    if feedback:
        st.write(f"Thank you for your feedback: {feedback}")
    return feedback

def main():
    st.title("Document Assistant")

    # User input
    user_input = st.text_input("Enter your question:", "")

    if user_input:
        with st.spinner("Generating response..."):
            try:
                # Initialize the language model
                llm = ChatOllama(model=MODEL_NAME)

                # Load the vector database
                vector_db = load_vector_db()
                if vector_db is None:
                    st.error("Failed to load or create the vector database.")
                    return

                # Suggest similar queries
                suggestions = suggest_similar_questions(user_input, vector_db, llm)
                st.write("**Suggested queries:**")
                for suggestion in suggestions:
                    st.write(f"- {suggestion}")

                # Create the retriever
                retriever = create_retriever(vector_db, llm)

                # Create the chain
                chain = create_chain(retriever, llm)

                # Get the response
                response = chain.invoke(input=user_input)

                st.markdown("**Assistant:**")
                st.write(response)

                # Capture feedback
                capture_feedback()

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    else:
        st.info("Please enter a question to get started.")

if __name__ == "__main__":
    main()
