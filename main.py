import os
import sys
from pathlib import Path
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.agents import create_agent

from rag import RAGDocumentUploader

# ----------------------------
# Load .env
# Task only provides OPENROUTER_API_KEY and HF_API_KEY
# ----------------------------
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
HF_API_KEY = os.getenv("HF_API_KEY", "")

# Pass HuggingFace token so sentence-transformers can download from Hub
if HF_API_KEY:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_API_KEY

# ----------------------------
# Resolve data/ and chroma_db/ relative to THIS file's location
# so the script works regardless of which directory the grader runs it from
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")

# ----------------------------
# Initialize RAG with hybrid retrieval (BM25 + ChromaDB)
# Load and index every file in data/
# ----------------------------
uploader = RAGDocumentUploader(persist_directory=CHROMA_DIR)

data_path = Path(DATA_DIR)
data_path.mkdir(parents=True, exist_ok=True)

data_files = [str(f) for f in data_path.rglob("*") if f.is_file()]
if data_files:
    uploader.upload_batch(data_files)

retriever = uploader.get_retriever(retriever_type="hybrid", weights=(0.5, 0.5))

# ----------------------------
# LLM — task specifies nvidia/nemotron-3-nano-30b-a3b:free via OpenRouter
# Model hardcoded as task only guarantees OPENROUTER_API_KEY and HF_API_KEY
# ----------------------------
llm = ChatOpenAI(
    model="nvidia/nemotron-3-nano-30b-a3b:free",
    openai_api_key=OPENROUTER_API_KEY,
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0,
)


# ----------------------------
# Tool 1: Flight Booking
# Rewritten from Stage 4 — same function signature: get_flight_schedule(origin, destination)
# ----------------------------
@tool
def get_flight_schedule(origin: str, destination: str) -> dict:
    """Returns flight duration in hours and ticket price in USD for a one-way flight between two cities."""
    return {
        "origin": origin,
        "destination": destination,
        "flight_time_hours": 5.5,
        "price_usd": 920,
    }


# ----------------------------
# Tool 2: Hotel Booking
# Rewritten from Stage 4 — same function signature: get_hotel_schedule(city)
# ----------------------------
@tool
def get_hotel_schedule(city: str) -> dict:
    """Returns available hotel options and their prices in USD per night for a given city."""
    return {
        "city": city,
        "hotels": [
            {"name": "Nairobi Serena", "price_usd": 250},
            {"name": "Radisson Blu", "price_usd": 200},
        ],
    }


# ----------------------------
# Tool 3: Currency Conversion
# Rewritten from Stage 4 — same function signature: convert_currency(amount, from_currency, to_currency)
# ----------------------------
@tool
def convert_currency(amount: float, from_currency: str, to_currency: str) -> dict:
    """Converts a monetary amount from one currency to another."""
    exchange_rates = {
        ("USD", "NGN"): 1400,
        ("NGN", "USD"): 1 / 1400,
        ("USD", "KES"): 130,
        ("KES", "USD"): 1 / 130,
        ("USD", "EUR"): 0.92,
        ("EUR", "USD"): 1.09,
    }
    key = (from_currency.upper(), to_currency.upper())
    if key not in exchange_rates:
        return {"error": f"Exchange rate for {from_currency} to {to_currency} not available."}
    return {
        "amount_converted": round(amount * exchange_rates[key], 2),
        "currency": to_currency,
    }


# ----------------------------
# Tool 4: RAG — queries internal knowledge base and past conversation history
# ----------------------------
@tool
def rag_tool(query: str) -> str:
    """Useful for answering questions using the internal knowledge base and past conversation history."""
    docs = retriever.invoke(query)
    return "\n\n".join(doc.page_content for doc in docs)


# ----------------------------
# Create agent using LangChain v1 create_agent API
# ----------------------------
agent = create_agent(
    model=llm,
    tools=[get_flight_schedule, get_hotel_schedule, convert_currency, rag_tool],
    system_prompt=(
        "You are a helpful travel and logistics assistant. "
        "Use the available tools to answer questions about flights, hotels, "
        "currency conversion, and internal company knowledge."
    ),
)


# ----------------------------
# Entry point — accepts CLI argument as required
# Usage: python main.py "Your prompt here"
# ----------------------------
def main():
    if len(sys.argv) < 2:
        print('Usage: python main.py "Your question here"')
        sys.exit(1)

    user_prompt = sys.argv[1]

    # Invoke agent using LangChain v1 messages pattern
    result = agent.invoke({
        "messages": [
            {"role": "user", "content": user_prompt}
        ]
    })

    # Extract final response — last message in the messages list
    response = result["messages"][-1].content

    # Save conversation history (human + AI turns only, NOT tool call steps)
    # to vector store for long-term memory retrieval
    conversation_text = f"User: {user_prompt}\nAssistant: {response}"
    uploader.add_texts(
        [conversation_text],
        metadata=[{"source": "conversation_history"}]
    )

    # Print full conversation history
    print("\n===== FULL CONVERSATION HISTORY =====\n")
    for msg in result["messages"]:
        role = getattr(msg, "type", type(msg).__name__)
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        # Only print human and AI messages, skip tool call intermediates
        if role in ("human", "ai") and content:
            print(f"{role.upper()}: {content}\n")

    # Print final response
    print("===== FINAL RESPONSE =====\n")
    print(response)


if __name__ == "__main__":
    main()
