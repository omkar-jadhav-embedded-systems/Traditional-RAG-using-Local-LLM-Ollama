# ui.py

import streamlit as st
from app import UniversalDocumentLoader, DocumentChunker, EmbeddingManager, VectorStore, LocalLLM

# =========================================================
# STREAMLIT UI & CACHING
# =========================================================

# Set the page configuration
st.set_page_config(
    page_title="Local RAG Chatbot",
    page_icon="🧠",
    layout="wide"
)


# Decorator to cache the setup process, so it runs only once.
@st.cache_resource
def setup_rag_pipeline():
    """
    Loads documents, chunks them, generates embeddings, and stores them in ChromaDB.
    This entire process is cached and will only run once.
    """
    with st.spinner("Setting up the RAG pipeline... Please wait."):
        # STEP 1: Load documents
        loader = UniversalDocumentLoader("./Data")
        documents = loader.load_all_documents()

        if not documents:
            st.error("No documents found in the '/Data' directory. Please add some files to continue.")
            st.stop()

        # STEP 2: Chunk documents
        chunker = DocumentChunker()
        chunks = chunker.chunk_documents(documents)

        # STEP 3: Generate embeddings and store in ChromaDB
        embedding_manager = EmbeddingManager()
        vector_store = VectorStore()

        # Avoid duplicate insertion
        if vector_store.collection.count() == 0:
            st.write("Building the vector store for the first time...")
            chunk_texts = [chunk.page_content for chunk in chunks]
            embeddings = embedding_manager.generate_embeddings(chunk_texts)
            vector_store.add_documents(chunks, embeddings)
            st.success("Vector store built successfully!")
        else:
            st.write("Vector store already exists.")

    return vector_store, embedding_manager, LocalLLM()


# --- Main Streamlit App ---

st.title("🧠 Chat With Your Documents")
st.caption("This chatbot is powered by local models, ensuring your data remains private.")

# Setup the RAG pipeline and get the necessary components
try:
    vector_store, embedding_manager, llm = setup_rag_pipeline()
except Exception as e:
    st.error(f"Failed to set up the RAG pipeline: {e}")
    st.stop()

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "How can I help you today?"}]

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        with st.spinner("Thinking..."):
            # Generate query embedding
            query_embedding = embedding_manager.generate_embeddings([prompt])[0]

            # Retrieve top chunks
            results = vector_store.similarity_search(query_embedding, top_k=5)
            retrieved_docs = results["documents"][0]

            # Generate response
            full_response = llm.generate_response(prompt, retrieved_docs)
            message_placeholder.markdown(full_response)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
