import os
from crewai import Agent, Task, Crew
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

# Initialize Groq LLM
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.1-8b-instant"
)

def run_crew(hr_text):
    """
    Runs the CrewAI pipeline to generate an AI response
    to the HR spoken input received from Twilio.
    """

    # Define the AI agent
    assistant_agent = Agent(
        role="AI HR Interview Assistant",
        goal="Respond politely, professionally and clearly to HR questions.",
        backstory=(
            "You assist job applicants by giving short, clear and friendly replies "
            "during HR screening calls. Keep your tone natural and confident."
        ),
        llm=llm
    )

    # Create Task
    task = Task(
        description=f"HR said: '{hr_text}'. Give a clear and polite response.",
        agent=assistant_agent
    )

    # Create Crew
    crew = Crew(
        agents=[assistant_agent],
        tasks=[task]
    )

    # Run pipeline (new CrewAI v1.x method)
    result = crew.kickoff()

    # Return raw text output
    return result.raw_output
