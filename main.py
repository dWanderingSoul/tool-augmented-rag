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
HF_API_KEY = os.getenv("HF_API_KEY", "")

# ----------------------------
# Initialize RAG
# ----------------------------
rag = RAGSystem(data_path="data")


# ----------------------------
# Tool 1: Flight Booking
# FIX: Returns Lagos <-> Nairobi data (matching the Stage 4 reference tool),
#      not Lagos-Abuja/London which was irrelevant to the task prompt.
# ----------------------------
def get_flight_schedule(query: str) -> str:
    return (
        "Flight Schedule (USD, one-way):\n"
        "- Lagos (LOS) → Nairobi (NBO): flight time 5.5 hours, price $920\n"
        "- Nairobi (NBO) → Lagos (LOS): flight time 5.5 hours, price $920\n"
        "Round-trip total: 11 hours flight time, $1,840 USD"
    )


# ----------------------------
# Tool 2: Hotel Booking
# FIX: Returns Nairobi hotels (matching the Stage 4 reference tool values).
#      Original returned Lagos/Abuja hotels — wrong city for the prompt.
# ----------------------------
def get_hotel_schedule(query: str) -> str:
    return (
        "Hotel options in Nairobi (USD per night):\n"
        "- Nairobi Serena Hotel: $250/night\n"
        "- Radisson Blu Nairobi: $200/night\n"
        "3-night stay costs: Nairobi Serena $750 | Radisson Blu $600"
    )


# ----------------------------
# Tool 3: Currency Conversion
# ----------------------------
def convert_currency(query: str) -> str:
    return (
        "Currency conversion rates:\n"
        "- 1 USD = 1,400 NGN (Nigerian Naira)\n"
        "- 1 USD = 130 KES (Kenyan Shilling)\n"
        "- 1 USD = 0.92 EUR\n"
        "Example: $2,440 USD = 3,416,000 NGN"
    )


# ----------------------------
# Tool 4: RAG (internal knowledge)
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
        description=(
            "Use this to get flight schedule, flight duration in hours, "
            "and ticket pricing in USD between cities."
        )
    ),
    Tool(
        name="HotelBookingTool",
        func=get_hotel_schedule,
        description=(
            "Use this to get hotel booking options and nightly prices in USD "
            "for a destination city."
        )
    ),
    Tool(
        name="CurrencyConversionTool",
        func=convert_currency,
        description=(
            "Use this to convert currency amounts between different currencies "
            "such as USD, NGN (Nigerian Naira), KES (Kenyan Shilling), EUR."
        )
    ),
    Tool(
        name="InternalKnowledgeRAGTool",
        func=query_rag,
        description=(
            "Use this to retrieve internal company knowledge, documents, "
            "policies, and past conversation history from the knowledge base."
        )
    )
]

# ----------------------------
# LLM Setup
#      Task specifies nvidia/nemotron-3-nano-30b-a3b:free.
#      Now reads from LLM_MODEL_NAME env var with that as default.
# ----------------------------
llm = ChatOpenAI(
    openai_api_key=OPENROUTER_API_KEY,
    openai_api_base="https://openrouter.ai/api/v1",
    model=os.getenv("LLM_MODEL_NAME", "nvidia/nemotron-3-nano-30b-a3b:free")
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
    verbose=True   #  verbose=True so tool call chain is visible in stdout
)


# ----------------------------
# Main
# ----------------------------
def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py \"Your question here\"")
        sys.exit(1)

    user_prompt = sys.argv[1]

    print(f"\n>>> User: {user_prompt}\n")

    response = agent.run(user_prompt)

    # Save conversation into vector store for long-term memory retention
    conversation_text = f"User: {user_prompt}\nAssistant: {response}"
    rag.save_conversation(conversation_text)

    # Print full conversation history as required by the task
    print("\n===== FULL CONVERSATION HISTORY =====\n")
    for msg in memory.chat_memory.messages:
        role = msg.type.upper()
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        print(f"{role}: {content}\n")

    print("===== FINAL RESPONSE =====\n")
    print(response)


if __name__ == "__main__":
    main()
