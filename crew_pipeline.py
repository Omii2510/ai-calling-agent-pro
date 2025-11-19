import os
from crewai import Agent, Task, Crew
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# Load .env values
load_dotenv()

# Disable CrewAI from trying to load OpenAI automatically
os.environ["OPENAI_API_KEY"] = "disabled"

# Initialize Groq LLM
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.1-8b-instant"
)

def run_crew(hr_text):
    """
    Runs the CrewAI pipeline to generate a response
    to the HR messages captured from Twilio's call.
    """

    # Agent Definition
    assistant_agent = Agent(
        role="AI HR Interview Assistant",
        goal="Provide short, polite and confident responses to HR questions.",
        backstory=(
            "You assist candidates during HR screening calls by giving "
            "clear and helpful responses when HR speaks."
        ),
        llm=llm
    )

    # Task Definition
    task = Task(
        description=f"HR said: '{hr_text}'. Respond politely and naturally.",
        agent=assistant_agent
    )

    # Crew Setup
    crew = Crew(
        agents=[assistant_agent],
        tasks=[task]
    )

    # Run CrewAI (v1.x system)
    result = crew.kickoff()

    # Return raw LLM text
    return result.raw_output
