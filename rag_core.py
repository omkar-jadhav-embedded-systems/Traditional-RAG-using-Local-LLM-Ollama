# =========================================================
# COMPLETE LOCAL RAG PIPELINE
# =========================================================
import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader, PyMuPDFLoader, CSVLoader, JSONLoader, Docx2txtLoader, UnstructuredMarkdownLoader, BSHTMLLoader, UnstructuredPowerPointLoader, UnstructuredExcelLoader
import numpy as np
import chromadb
import uuid
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import ollama

# =========================================================
# UNIVERSAL DOCUMENT LOADER
# =========================================================

class UniversalDocumentLoader:

    def __init__(self, data_directory: str = "./Data"):

        self.data_directory = data_directory

        self.documents = []

    # -----------------------------------------------------
    # SAFE LOADING FUNCTION
    # -----------------------------------------------------

    def _safe_load(self, loader, file_type: str):

        try:

            docs = loader.load()

            print(f"Loaded {len(docs)} {file_type} documents")

            self.documents.extend(docs)

        except Exception as e:

            print(f"Error loading {file_type}: {e}")

    def load_all_documents(self):
        # TXT FILES
        txt_loader = DirectoryLoader(
            self.data_directory,
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
            show_progress=True
        )

        self._safe_load(txt_loader, "TXT")
        pdf_loader = DirectoryLoader(
            self.data_directory,
            glob="**/*.pdf",
            loader_cls = PyMuPDFLoader,
            show_progress=True
        )

        self._safe_load(pdf_loader, "PDF")


        # DOCX FILES
        docx_loader = DirectoryLoader(
            self.data_directory,
            glob="**/*.docx",
            loader_cls = Docx2txtLoader,
            show_progress=True
        )

        self._safe_load(docx_loader, "DOCX")


        # CSV FILES
        csv_loader = DirectoryLoader(
            self.data_directory,
            glob="**/*.csv",
            loader_cls=CSVLoader,
            show_progress=True
        )

        self._safe_load(csv_loader, "CSV")


        # MARKDOWN FILES
        md_loader = DirectoryLoader(
            self.data_directory,
            glob="**/*.md",
            loader_cls=UnstructuredMarkdownLoader,
            show_progress=True
        )

        self._safe_load(md_loader, "MARKDOWN")


        # HTML FILES
        html_loader = DirectoryLoader(
            self.data_directory,
            glob="**/*.html",
            loader_cls=BSHTMLLoader,
            show_progress=True
        )

        self._safe_load(html_loader, "HTML")


        # POWERPOINT FILES
        ppt_loader = DirectoryLoader(
            self.data_directory,
            glob="**/*.pptx",
            loader_cls=UnstructuredPowerPointLoader,
            show_progress=True
        )

        self._safe_load(ppt_loader, "PPTX")


        # EXCEL FILES
        excel_loader = DirectoryLoader(
            self.data_directory,
            glob="**/*.xlsx",
            loader_cls=UnstructuredExcelLoader,
            show_progress=True
        )

        self._safe_load(excel_loader, "EXCEL")


        print("\n========================================")
        print(f"TOTAL DOCUMENTS LOADED: {len(self.documents)}")
        print("========================================")

        return self.documents


# =========================================================
# DOCUMENT CHUNKER
# =========================================================

class DocumentChunker:
    def __init__(self, chunk_size: int = 1200, chunk_overlap: int = 250):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = None
        self._initialize_chunker()

    def _initialize_chunker(self):
        try:
            self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap, length_function=len)

            print("Document chunker initialized successfully.")
            print(f"Chunk Size: {self.chunk_size}")
            print(f"Chunk Overlap: {self.chunk_overlap}")

        except Exception as e:
            print(f"Error initializing chunker: {e}")
            raise

    def chunk_documents(self, documents: List[Any]):
        try:
            chunks = self.text_splitter.split_documents(documents)

            print(f"Original documents count: {len(documents)}")
            print(f"Generated chunks count: {len(chunks)}")

            return chunks

        except Exception as e:
            print(f"Error while chunking documents: {e}")
            raise


# =========================================================
# EMBEDDING MANAGER
# =========================================================

class EmbeddingManager:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        try:
            print(f"Loading embedding model {self.model_name}...")
            self.model = SentenceTransformer(self.model_name)
            print(f"Model loaded successfully. Embedding dimension {self.model.get_embedding_dimension()}")
        except Exception as e:
            print(f"Error loading model: {self.model_name} : {e}")
            raise

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not loaded.")
        try:
            print(f"Generating embeddings for {len(texts)} texts...")
            embeddings = self.model.encode(texts, show_progress_bar=True)
            print(f"Generated embeddings with shape: {embeddings.shape}")
            return embeddings
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            raise


# =========================================================
# VECTOR STORE
# =========================================================

class VectorStore:
    def __init__(self, collection_name: str = "document_collection", persist_directory: str = "./chroma_db"):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self._initialize_store()

    def _initialize_store(self):
        try:
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory) #saves permanently to disk the vector database
            self.collection = self.client.get_or_create_collection(name=self.collection_name, metadata={"description":"PDF document embeddings for RAG"})
            print(f"Vector store initialized with collection: {self.collection_name}")
            print(f"Existing documents in collection: {self.collection.count()}")

        except Exception as e:
            print(f"Error initializing vector store: {e}")
            raise

    def add_documents(self, documents: List[Any], embeddings: np.ndarray): #Here documents is the list of chunks after text splitting.
        if len(documents) != len(embeddings):   #Checking the number of vectors are equal to number of chunks of text.
            raise ValueError("Number of documents and embeddings must match.")

        ids = []
        metadatas=[]
        documents_text=[]
        embeddings_list=[]

        for i, (doc, embedding) in enumerate(zip(documents, embeddings)): #zip() combines horizontally and enumerate() adds numbering [0 → (doc1, vec1)],
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"  #Unique ID for each document chunk first 8 characters of the unique identifier which is hex of the uuid and the index of the document chunk in the list of documents.
            ids.append(doc_id)
            """Actual structure of the document in the vector store will be like this:
                {
                    "id": "doc_a91f82bc_0",
                
                    "embedding": [0.12, -0.44, 0.98, ...],
                
                    "document": "Python is a programming language...",
                
                    "metadata": {
                        "source": "./Data/python.txt",
                        "doc_index": 0,
                        "content_length": 1200
                        }
                }"""
            metadata = dict(doc.metadata)
            metadata['doc_index'] = i
            metadata['content_length'] = len(doc.page_content)
            metadatas.append(metadata)

            documents_text.append(doc.page_content)

            embeddings_list.append(embedding.tolist())

        try:
            self.collection.add(ids=ids, metadatas=metadatas, documents=documents_text, embeddings=embeddings_list)
            print(f"Added {len(documents)} documents to vector store.")
            print(f"Successfully added {len(documents)} documents to vector store.")
            print(f"Total documents in collection after addition: {self.collection.count()}")

        except Exception as e:
            print(f"Error adding documents to vector store: {e}")
            raise

    def similarity_search(self,query_embedding, top_k: int = 3):

        results = self.collection.query(query_embeddings=[query_embedding.tolist()], n_results=top_k)

        return results


class LocalLLM:

    def __init__(
        self,
        model_name: str = "qwen2.5-coder:14b"
    ):

        self.model_name = model_name

        print(f"Using local LLM: {self.model_name}")

    def generate_response(self, query, retrieved_docs):
        context = "\n\n".join(retrieved_docs)

        response = ollama.chat(
            model=self.model_name,
            messages=[
                {
                    "role": "system",
                    "content": """
    You are an intelligent document question-answering assistant.

    Answer the user's question ONLY using the provided context.

    Rules:
    1. Do not invent information.
    2. Preserve important numbers, limits, names, and values exactly.
    3. Keep technical accuracy.
    4. Use bullet points when helpful.
    5. If the answer is not found in the context, say:
       "Answer not found in the provided documents."
    6. Keep the answer concise but complete.
    7. Include all exact numerical values from context.
    8. Never summarize away limits, prices, percentages, dates, or counts.
    9. If multiple relevant entries exist, include ALL of them.
    10. Structure answer clearly using bullet points.
    """
                },
                {
                    "role": "user",
                    "content": f"""
    ================ CONTEXT ================

    {context}

    =========================================

    QUESTION:
    {query}
    """
                }
            ]
        )

        return response["message"]["content"]

# =========================================================
# TRADITIONAL RAG PIPELINE
# =========================================================

def main():
    # -----------------------------------------------------
    # STEP 1 : LOAD DOCUMENTS
    # -----------------------------------------------------

    loader = UniversalDocumentLoader("./Data")

    documents = loader.load_all_documents()

    # -----------------------------------------
    # STEP 2 : CHUNK DOCUMENTS
    # -----------------------------------------
    chunker = DocumentChunker()
    chunks = chunker.chunk_documents(documents)
    print(f"Total chunks created: {len(chunks)}")
    print(f"Sample chunk content: {chunks[0].page_content[:500]}...")  # Print the first 500 characters of the first chunk

    # -----------------------------------------
    # STEP 3 : GENERATE EMBEDDINGS
    # -----------------------------------------

    embedding_manager = EmbeddingManager()

    chunk_texts = [chunk.page_content for chunk in chunks]  # Extracting the text content from each chunk to generate embeddings
    #print(f"chunk_texts={chunk_texts}")  # Print the list of chunk texts to verify the content before generating embeddings
    embeddings = embedding_manager.generate_embeddings(chunk_texts) #Generating embeddings for the chunked documents using the embedding manager. The generate_embeddings method takes a list of texts and returns their corresponding embeddings as a numpy array.

    print(f"Embeddings generated for {len(chunk_texts)} chunks.")
    print(f"Number of Embeddings: {len(embeddings)}")

    # -----------------------------------------
    # STEP 4 : STORE IN CHROMADB
    # -----------------------------------------

    vector_store = VectorStore()

    # Avoid duplicate insertion
    if vector_store.collection.count() == 0:
        vector_store.add_documents(chunks, embeddings)
    else:
        print("\nDocuments already exist in ChromaDB.")

    # -----------------------------------------
    # STEP 5 : QUERY LOOP
    # -----------------------------------------
    llm = LocalLLM()
    while True:

        query = input("\nEnter your query (or type exit): ")

        if query.lower() == "exit":
            break

        # -------------------------------------
        # Generate query embedding
        # -------------------------------------

        query_embedding = embedding_manager.generate_embeddings([query])[0]
        print(f"query_embedding={query_embedding}")  # Print the query embedding to verify its content before performing similarity search
        # -------------------------------------
        # Retrieve top chunks
        # -------------------------------------

        results = vector_store.similarity_search(query_embedding, top_k=5)
        print(f"Similarity search results: {results}")  # Print the raw results from the similarity search to verify the retrieved documents, metadata, and distances
        retrieved_docs = results["documents"][0]
        retrieved_metadata = results["metadatas"][0]
        distances = results["distances"][0]

        print("\n================ RETRIEVED CHUNKS ================\n")

        for i, (doc, meta, distance) in enumerate(
            zip(retrieved_docs, retrieved_metadata, distances)
        ):

            print(f"Result #{i+1}")
            print(f"Distance Score: {distance}")

            print(f"\nMetadata:")
            print(meta)

            print(f"\nContent:\n{doc[:1000]}")

            print("\n--------------------------------------------------\n")

        # -----------------------------------------
        # STEP 6 : LOCAL LLM RESPONSE GENERATION
        # -----------------------------------------

        response = llm.generate_response(query, retrieved_docs)

        print("\n================ LLM RESPONSE ================\n")
        print(response)
        print("\n=============================================\n")


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":
    main()
