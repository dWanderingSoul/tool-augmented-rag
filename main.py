import os
import sys
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory

from rag import RAGSystem

# ----------------------------
# Load .env
# ----------------------------
load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
HF_API_KEY = os.getenv("HF_API_KEY")

# ----------------------------
# Initialize RAG
# ----------------------------
rag = RAGSystem(data_path="data")

# ----------------------------
# Tool 1
# ----------------------------
def get_flight_schedule(query: str) -> str:
    return """
Flights:
- Lagos → Abuja | $120
- Abuja → London | $850
"""

# ----------------------------
# Tool 2
# ----------------------------
def get_hotel_schedule(query: str) -> str:
    return """
Hotels:
- Abuja Grand Hotel | $150/night
- Lagos Continental | $220/night
"""

# ----------------------------
# Tool 3
# ----------------------------
def convert_currency(query: str) -> str:
    return """
1 USD = 1500 NGN
1 USD = 0.92 EUR
"""

# ----------------------------
# Tool 4 (RAG)
# ----------------------------
def query_rag(query: str) -> str:
    return rag.query(query)

# ----------------------------
# Register Tools
# ----------------------------
tools = [
    Tool(
        name="FlightBookingTool",
        func=get_flight_schedule,
        description="Returns flight schedule with pricing in USD."
    ),
    Tool(
        name="HotelBookingTool",
        func=get_hotel_schedule,
        description="Returns hotel booking schedule with pricing in USD."
    ),
    Tool(
        name="CurrencyConversionTool",
        func=convert_currency,
        description="Converts currency."
    ),
    Tool(
        name="InternalKnowledgeRAGTool",
        func=query_rag,
        description="Queries internal company knowledge from RAG."
    )
]

# ----------------------------
# LLM Setup
# ----------------------------
llm = ChatOpenAI(
    openai_api_key=OPENROUTER_API_KEY,
    openai_api_base="https://openrouter.ai/api/v1",
    model="openai/gpt-4o-mini"
)

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.OPENAI_FUNCTIONS,
    memory=memory,
    verbose=False
)

# ----------------------------
# Main
# ----------------------------
def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py \"Your question here\"")
        sys.exit(1)

    user_prompt = sys.argv[1]

    response = agent.run(user_prompt)

    # Save conversation into vector store
    conversation_text = f"User: {user_prompt}\nAssistant: {response}"
    rag.save_conversation(conversation_text)

    print("\n===== FULL CONVERSATION HISTORY =====\n")
    for msg in memory.chat_memory.messages:
        print(f"{msg.type.upper()}: {msg.content}")
        print()

    print("===== FINAL RESPONSE =====\n")
    print(response)


if __name__ == "__main__":
    main()
