import os
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document


class RAGSystem:
    def __init__(self, data_path: str = "data", persist_directory: str = "./chroma_db"):
        self.data_path = data_path
        self.persist_directory = persist_directory

        #  Use the exact embedding model specified in the task
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # Load and index all documents from data/
        documents = self._load_documents()

        if documents:
            #  Use ChromaDB (task requirement), not FAISS
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
        else:
            # Gracefully handle empty data/ directory without crashing
            self.vectorstore = Chroma(
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory
            )

    def _load_documents(self):
        """Load all files from the data/ folder and split into chunks."""
        # Create data/ if it doesn't exist so the script never crashes on missing folder
        if not os.path.exists(self.data_path):
            os.makedirs(self.data_path)
            return []

        try:
            loader = DirectoryLoader(
                self.data_path,
                glob="**/*.*",
                loader_cls=TextLoader,
                show_progress=False,   #  suppress noise from stdout
                silent_errors=True     # skip binary/unreadable files gracefully
            )
            raw_docs = loader.load()
        except Exception as e:
            print(f"[RAG] Warning: Could not load documents: {e}")
            return []

        if not raw_docs:
            return []

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        return splitter.split_documents(raw_docs)

    def query(self, query: str, k: int = 3) -> str:
        """Semantic similarity search over the vector store."""
        try:
            results = self.vectorstore.similarity_search(query, k=k)
            if not results:
                return "No relevant information found in the internal knowledge base."
            return "\n\n".join([doc.page_content for doc in results])
        except Exception as e:
            return f"RAG query error: {e}"

    def save_conversation(self, conversation_text: str):
        """
        Persist a conversation turn into the vector store.
        This satisfies the requirement: 'conversation history is also saved
        in the vector store so it can be retrieved at a later time'.
        """
        try:
            self.vectorstore.add_documents([
                Document(
                    page_content=conversation_text,
                    metadata={"source": "conversation_history"}
                )
            ])
        except Exception as e:
            print(f"[RAG] Warning: Could not save conversation: {e}")
