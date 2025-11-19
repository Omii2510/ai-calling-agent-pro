import os
from crewai import Agent, Task, Crew
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

# Disable all OpenAI usage inside CrewAI
os.environ["OPENAI_API_KEY"] = ""
os.environ["OPENAI_ORGANIZATION"] = ""
os.environ["OPENAI_BASE_URL"] = ""
os.environ["CREWAI_DISABLE_OPENAI"] = "1"

# Initialize Groq LLM
llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.1-8b-instant"
)

def run_crew(hr_text: str) -> str:
    """
    Generate a short, polite HR reply using Groq + CrewAI.
    """

    assistant_agent = Agent(
        role="AI HR Interview Assistant",
        goal="Give short, polite, professional replies.",
        backstory=(
            "You help candidates during HR screening calls. "
            "Responses must be 1–2 polite sentences."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
        tools=[]  # IMPORTANT → prevents OpenAI tools
    )

    task = Task(
        description=f"HR said: '{hr_text}'. Respond politely.",
        expected_output="A short, friendly 1–2 sentence reply.",
        agent=assistant_agent
    )

    crew = Crew(
        agents=[assistant_agent],
        tasks=[task],
        verbose=False
    )

    result = crew.kickoff()

    try:
        return result.raw_output
    except:
        return str(result)
