import os
from crewai import Agent, Task, Crew
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Disable OpenAI fallback
os.environ["OPENAI_API_KEY"] = "disabled"

# Initialize Groq LLM
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.1-8b-instant"
)

def run_crew(hr_text):
    """
    Process HR audio text and generate an AI response using CrewAI.
    """

    # Agent definition
    assistant_agent = Agent(
        role="AI HR Interview Assistant",
        goal="Respond politely, professionally, and clearly to HR questions.",
        backstory=(
            "You help candidates during HR screening calls. "
            "You speak clearly, politely and keep responses short."
        ),
        llm=llm
    )

    # Task (CrewAI 1.x requires expected_output)
    task = Task(
        description=f"HR said: '{hr_text}'. Respond politely and naturally.",
        expected_output="A short, polite spoken reply (1–2 sentences).",
        agent=assistant_agent
    )

    # Create Crew
    crew = Crew(
        agents=[assistant_agent],
        tasks=[task]
    )

    # Run CrewAI
    result = crew.kickoff()

    return result.raw_output
