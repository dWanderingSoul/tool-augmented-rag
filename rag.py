import os
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document


class RAGSystem:
    def __init__(self, data_path="data"):
        self.data_path = data_path

        # Load all files from data/
        loader = DirectoryLoader(
            self.data_path,
            glob="**/*",
            loader_cls=TextLoader,
            show_progress=True
        )

        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

        docs = text_splitter.split_documents(documents)

        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        self.vectorstore = FAISS.from_documents(docs, self.embeddings)

    def query(self, query: str):
        results = self.vectorstore.similarity_search(query, k=3)
        return "\n\n".join([doc.page_content for doc in results])

    def save_conversation(self, conversation_text: str):
        self.vectorstore.add_documents([
            Document(page_content=conversation_text)
        ])
