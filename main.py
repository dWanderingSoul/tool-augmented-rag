import os
import sys
from pathlib import Path
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.tools import tool
from langchain.agents import initialize_agent, AgentType
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    CSVLoader,
    Docx2txtLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever


# =========================
# LOAD ENVIRONMENT
# =========================
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
HF_API_KEY = os.getenv("HF_API_KEY")

if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY not found in .env")

if not HF_API_KEY:
    raise ValueError("HF_API_KEY not found in .env")

os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_API_KEY


# =========================
# LOAD DOCUMENTS FROM data/
# =========================
def load_documents_from_folder(folder_path="data"):
    documents = []
    folder = Path(folder_path)

    if not folder.exists():
        folder.mkdir()

    for file_path in folder.glob("*"):
        if file_path.suffix == ".txt":
            loader = TextLoader(str(file_path), encoding="utf-8")
        elif file_path.suffix == ".pdf":
            loader = PyPDFLoader(str(file_path))
        elif file_path.suffix == ".csv":
            loader = CSVLoader(str(file_path))
        elif file_path.suffix == ".docx":
            loader = Docx2txtLoader(str(file_path))
        else:
            continue

        documents.extend(loader.load())

    return documents


# =========================
# BUILD VECTOR STORE
# =========================
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

raw_docs = load_documents_from_folder("data")

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(raw_docs)

vectorstore = Chroma.from_documents(
    docs,
    embeddings,
    persist_directory="./chroma_db"
)

vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
bm25_retriever = BM25Retriever.from_documents(docs)
bm25_retriever.k = 4

hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.5, 0.5],
)


# =========================
# MEMORY (stored in vector DB)
# =========================
def save_conversation(user_input, assistant_output):
    memory_doc = Document(
        page_content=f"User: {user_input}\nAssistant: {assistant_output}"
    )
    vectorstore.add_documents([memory_doc])


def get_full_history():
    history_docs = vectorstore.similarity_search("User:", k=20)
    return "\n\n".join([doc.page_content for doc in history_docs])


# =========================
# TOOL DEFINITIONS
# =========================
@tool
def get_flight_schedule(origin: str, destination: str) -> str:
    """Returns flight schedule with pricing in USD."""
    flights = [
        {"airline": "Air Peace", "price": 350},
        {"airline": "Qatar Airways", "price": 1200},
        {"airline": "Lufthansa", "price": 980},
    ]

    result = f"Flights from {origin} to {destination}:\n"
    for flight in flights:
        result += f"{flight['airline']} - ${flight['price']}\n"

    return result


@tool
def get_hotel_schedule(city: str) -> str:
    """Returns hotel booking schedule with pricing in USD."""
    hotels = [
        {"name": "Hilton", "price": 200},
        {"name": "Marriott", "price": 180},
        {"name": "Sheraton", "price": 150},
    ]

    result = f"Hotels in {city}:\n"
    for hotel in hotels:
        result += f"{hotel['name']} - ${hotel['price']} per night\n"

    return result


@tool
def convert_currency(amount: float, from_currency: str, to_currency: str) -> str:
    """Converts currency values."""
    rates = {
        "USD": 1,
        "NGN": 1500,
        "EUR": 0.9,
    }

    if from_currency not in rates or to_currency not in rates:
        return "Unsupported currency."

    usd_amount = amount / rates[from_currency]
    converted = usd_amount * rates[to_currency]

    return f"{amount} {from_currency} = {converted:.2f} {to_currency}"


@tool
def query_internal_knowledge(query: str) -> str:
    """Query internal RAG knowledge base."""
    docs = hybrid_retriever.invoke(query)
    if not docs:
        return "No relevant internal information found."

    return "\n\n".join([doc.page_content for doc in docs])


# =========================
# LLM (OpenRouter)
# =========================
llm = ChatOpenAI(
    model="nvidia/nemotron-3-nano-30b-a3b:free",
    openai_api_key=OPENROUTER_API_KEY,
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0,
)

tools = [
    get_flight_schedule,
    get_hotel_schedule,
    convert_currency,
    query_internal_knowledge,
]

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
)


# =========================
# CLI INPUT
# =========================
if len(sys.argv) > 1:
    user_prompt = sys.argv[1]
else:
    user_prompt = "Hello"


# =========================
# RUN AGENT
# =========================
response = agent.run(user_prompt)

# Save conversation
save_conversation(user_prompt, response)

# Retrieve full history
history = get_full_history()

# =========================
# PRINT OUTPUT
# =========================
print("\n========== FULL CONVERSATION HISTORY ==========\n")
print(history)

print("\n========== FINAL RESPONSE ==========\n")
print(response)
